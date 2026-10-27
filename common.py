"""Shared helpers: Ollama calls, image I/O, the cache of the command-line tools.

Only Ollama is contacted (http://localhost:11434 unless $METER_OCR_OLLAMA says otherwise); photos never leave the host.
"""
import base64, io, json, os, time, urllib.request
import numpy as np
from PIL import Image, ImageOps

OLLAMA = os.environ.get("METER_OCR_OLLAMA", "http://localhost:11434").rstrip("/") + "/api/chat"
# Working files of the step-by-step command-line tools (locate.py, crop.py, read.py, combine.py); the service
# keeps everything in memory.
CACHE = os.environ.get("METER_OCR_CACHE") or os.path.expanduser("~/.cache/meter-ocr")
LOCATOR = "qwen3-vl:8b-instruct"
KEEP_ALIVE = "30m"  # how long Ollama keeps a model loaded after a call; the service sets it short


def load_rgb(path):
    """Photo as an RGB numpy array, EXIF rotation applied."""
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def b64(arr, max_side=None, fmt="JPEG"):
    im = Image.fromarray(arr)
    if max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, fmt, **({"quality": 92} if fmt == "JPEG" else {}))
    return base64.b64encode(buf.getvalue()).decode()


def chat(model, prompt, images, fmt=None, num_predict=200, keep_alive=None):
    """One Ollama chat call; returns (text, seconds)."""
    body = {"model": model, "stream": False, "keep_alive": keep_alive or KEEP_ALIVE,
            "options": {"temperature": 0, "num_predict": num_predict},
            "messages": [{"role": "user", "content": prompt, "images": images}]}
    if fmt is not None:
        body["format"] = fmt
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.loads(r.read())["message"]["content"]
    return text, time.time() - t0


def cache_path(*parts):
    p = os.path.join(CACHE, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
