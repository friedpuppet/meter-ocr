"""meter-ocr: reads water-meter photos for Home Assistant, only while this host has been idle.

HA decides when to ask; this service only answers. Ollama stays on localhost; only this port needs to be open to HA.
  GET  /status       {"ready", "reason", "idle_min", "job"}
  POST /batch        multipart photos -> 202 {"id"}; 503 {"reason"} when the host is busy; 409 while a batch runs.
                     ?force=1 reads even when the host is busy (a person pressed a button); 409 still applies
  GET  /batch/<id>   {"state": "running"|"done"|"failed", "results": [...], "seconds", "error"}
Every request needs "Authorization: Bearer <token>".
Settings: /etc/meter-ocr/config.json ($METER_OCR_CONFIG). The token is generated on first start into
/var/lib/meter-ocr/token ($METER_OCR_STATE is the directory) unless the config sets one.
"""
import collections, email.parser, email.policy, glob, hmac, json, os, secrets, shutil, sys, tempfile, threading, time
import urllib.parse, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONFIG = os.environ.get("METER_OCR_CONFIG", "/etc/meter-ocr/config.json")
STATE = os.environ.get("METER_OCR_STATE", "/var/lib/meter-ocr")
DEFAULTS = {"listen": "0.0.0.0", "port": 8765, "gpu_max": 30, "cpu_max": 30, "idle_minutes": 10,
            "sample_seconds": 10, "gpu_busy": None, "token": None, "keep_alive": "1m", "max_photos": 20,
            "jobs_kept": 20}


def load_config():
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG):
        with open(CONFIG) as f:
            cfg.update(json.load(f))
    if not cfg["token"]:  # kept in the state directory; made on first start
        path = os.path.join(STATE, "token")
        if not os.path.exists(path):
            os.makedirs(STATE, exist_ok=True)
            with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
                f.write(secrets.token_urlsafe(32) + "\n")
        with open(path) as f:
            cfg["token"] = f.read().strip()
    if not cfg["gpu_busy"]:  # amdgpu exposes it; first card that has it
        found = sorted(glob.glob("/sys/class/drm/card*/device/gpu_busy_percent"))
        cfg["gpu_busy"] = found[0] if found else ""
    return cfg


def ensure_models(log=print):
    """Pull the models the pipeline uses if Ollama doesn't have them yet (first start after install)."""
    import common
    from pipeline import READERS
    base = common.OLLAMA.rsplit("/api/", 1)[0]
    for _ in range(60):  # Ollama may start after us
        try:
            with urllib.request.urlopen(base + "/api/tags", timeout=10) as r:
                have = {m["name"] for m in json.load(r)["models"]}
            break
        except OSError:
            time.sleep(10)
    else:
        log("Ollama doesn't answer; models not checked")
        return
    for model in (common.LOCATOR, *READERS):
        if model in have or model + ":latest" in have:
            continue
        log(f"pulling {model} ...")
        req = urllib.request.Request(base + "/api/pull", json.dumps({"model": model, "stream": False}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            log(f"{model}: {json.load(r).get('status')}")


class Load:
    """GPU and CPU busy % sampled every few seconds; ready once a full window is below both limits.

    Samples taken while our own batch runs are dropped: our load must not hold off the next batch.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = collections.deque()  # (time, gpu %, cpu %)
        self.own_work = threading.Event()
        self.lock = threading.Lock()

    @staticmethod
    def _cpu_times():
        with open("/proc/stat") as f:
            v = list(map(int, f.readline().split()[1:]))
        return sum(v), v[3] + v[4]  # total, idle + iowait

    def _gpu(self):
        try:
            with open(self.cfg["gpu_busy"]) as f:
                return float(f.read())
        except OSError:
            return 0.0

    def run(self):
        step = self.cfg["sample_seconds"]
        while True:
            # GPU busy % is instantaneous: average a few reads over the step; CPU from /proc/stat deltas
            t0, i0 = self._cpu_times()
            gpu = []
            for _ in range(5):
                gpu.append(self._gpu())
                time.sleep(step / 5)
            t1, i1 = self._cpu_times()
            cpu = 100.0 * (1 - (i1 - i0) / max(t1 - t0, 1))
            now = time.time()
            with self.lock:
                if self.own_work.is_set():
                    self.samples.clear()  # the window restarts after our own batch
                    continue
                if self.samples and now - self.samples[-1][0] > 3 * step:
                    self.samples.clear()  # a gap (sleep, suspend, paused): what came before says nothing about now
                self.samples.append((now, sum(gpu) / len(gpu), cpu))
                while self.samples and self.samples[0][0] < now - self.cfg["idle_minutes"] * 60:
                    self.samples.popleft()

    def state(self):
        """(ready, reason, idle minutes so far). Ready: samples cover the whole window, none over a limit."""
        with self.lock:
            s = list(self.samples)
        if self.own_work.is_set():
            return False, "читаю фото", 0.0
        now, window, step = time.time(), self.cfg["idle_minutes"] * 60, self.cfg["sample_seconds"]
        if not s:
            return False, "чекаю простою: ще немає замірів", 0.0
        over = [t for t, g, c in s if g >= self.cfg["gpu_max"] or c >= self.cfg["cpu_max"]]
        idle = round((now - (max(over) if over else s[0][0])) / 60, 1)
        if over and over[-1] == s[-1][0]:
            return False, f"ПК зайнятий: GPU {s[-1][1]:.0f} %, CPU {s[-1][2]:.0f} %", 0.0
        if over or s[0][0] > now - window + 2 * step:
            return False, f"чекаю простою: {idle} з {self.cfg['idle_minutes']} хв", idle
        return True, "готовий", idle


class Jobs:
    def __init__(self, cfg, load):
        self.cfg, self.load = cfg, load
        self.jobs = collections.OrderedDict()
        self.running = None
        self.lock = threading.Lock()

    def start(self, photos, tmp):
        with self.lock:
            if self.running:
                return None
            jid = secrets.token_hex(8)
            self.jobs[jid] = {"state": "running", "results": [], "started": time.time(), "files": [n for n, _ in photos]}
            while len(self.jobs) > self.cfg["jobs_kept"]:
                self.jobs.popitem(last=False)
            self.running = jid
        self.load.own_work.set()
        threading.Thread(target=self._run, args=(jid, photos, tmp), daemon=True).start()
        return jid

    def _run(self, jid, photos, tmp):
        job = self.jobs[jid]
        try:
            from pipeline import recognize  # heavy imports (OpenCV) only when needed
            job["results"] = recognize(photos, log=lambda m: print(f"[{jid}] {m}", flush=True))
            job["state"] = "done"
        except Exception as e:
            job["state"], job["error"] = "failed", repr(e)
            print(f"[{jid}] failed: {e!r}", file=sys.stderr, flush=True)
        finally:
            job["seconds"] = round(time.time() - job["started"], 1)
            shutil.rmtree(tmp, ignore_errors=True)
            with self.lock:
                self.running = None
            self.load.own_work.clear()


def handler(cfg, load, jobs):
    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self):
            got = self.headers.get("Authorization", "")
            if hmac.compare_digest(got.encode(), f"Bearer {cfg['token']}".encode()):
                return True
            self._send(401, {"error": "unauthorised"})
            return False

        def do_GET(self):
            if not self._authorised():
                return
            if self.path == "/status":
                ready, reason, idle = load.state()
                return self._send(200, {"ready": ready, "reason": reason, "idle_min": idle, "job": jobs.running})
            if self.path.startswith("/batch/"):
                job = jobs.jobs.get(self.path[len("/batch/"):])
                return self._send(200, job) if job else self._send(404, {"error": "no such batch"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorised():
                return
            url = urllib.parse.urlsplit(self.path)
            if url.path != "/batch":
                return self._send(404, {"error": "not found"})
            if jobs.running:
                return self._send(409, {"reason": "уже читаю інший пакет", "id": jobs.running})
            # ?force=1: read now even if the host isn't idle (a person asked for it); a running batch still wins
            force = urllib.parse.parse_qs(url.query).get("force", ["0"])[0].lower() in ("1", "true", "yes")
            ready, reason, _ = load.state()
            if not ready and not force:
                return self._send(503, {"reason": reason})
            length = int(self.headers.get("Content-Length", 0))
            msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
                b"Content-Type: " + self.headers.get("Content-Type", "").encode() + b"\r\n\r\n" + self.rfile.read(length))
            tmp = tempfile.mkdtemp(prefix="meter-ocr-")
            photos = []
            for part in msg.iter_parts():
                name = os.path.basename(part.get_filename() or "")
                if not name:
                    continue
                path = os.path.join(tmp, f"{len(photos):02d}-{name}")
                with open(path, "wb") as f:
                    f.write(part.get_payload(decode=True))
                photos.append((name, path))
            if not photos or len(photos) > cfg["max_photos"]:
                shutil.rmtree(tmp, ignore_errors=True)
                return self._send(400, {"error": f"expected 1..{cfg['max_photos']} photos, got {len(photos)}"})
            jid = jobs.start(photos, tmp)
            if jid is None:
                shutil.rmtree(tmp, ignore_errors=True)
                return self._send(409, {"reason": "уже читаю інший пакет", "id": jobs.running})
            self._send(202, {"id": jid, "photos": len(photos), "forced": force and not ready})

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return H


def main():
    cfg = load_config()
    import common
    common.KEEP_ALIVE = cfg["keep_alive"]  # free the GPU soon after a batch
    load = Load(cfg)
    threading.Thread(target=load.run, daemon=True).start()
    jobs = Jobs(cfg, load)
    threading.Thread(target=ensure_models, kwargs={"log": lambda m: print(m, flush=True)}, daemon=True).start()
    print(f"meter-ocr on {cfg['listen']}:{cfg['port']}, GPU load from {cfg['gpu_busy'] or 'nowhere'}", flush=True)
    ThreadingHTTPServer((cfg["listen"], cfg["port"]), handler(cfg, load, jobs)).serve_forever()


if __name__ == "__main__":
    main()
