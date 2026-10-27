"""OCR the crops from crop.py: counter window and serial number; digits with per-digit probability.

Models: glm-ocr / qwen3-vl via Ollama (token logprobs give the probability), tesseract (word confidence).
Filters (combine with '+'): none, invert, auto, up2, up3, sharp, gray, minch.
Output: <cache>/read/<model>__<filters>/<name>.json
"""
import argparse, base64, io, json, math, os, subprocess, sys, tempfile, time, urllib.request
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import common
from common import CACHE, OLLAMA, cache_path
from crop import PADS, PART_PADS

PROMPTS = {
    "glm-ocr": {"counter": "Text Recognition:", "serial": "Text Recognition:", "black": "Text Recognition:",
                "red": "Text Recognition:"},
    "deepseek-ocr": dict.fromkeys(["counter", "serial", "black", "red"], "Free OCR."),
    "qwen3-vl:8b-instruct": {
        "counter": "This is the digit counter of a water meter: 5 black wheels then 3 red wheels. "
                   "Write the 8 digits left to right, nothing else.",
        "serial": "Write the number in this image, digits only, nothing else.",
        "black": "Write the digits on the number wheels left to right, nothing else.",
        "red": "Write the digits on the number wheels left to right, nothing else."},
}


def apply(img, filters):
    for f in filters.split("+"):
        if f == "none":
            continue
        elif f == "invert":
            img = ImageOps.invert(img.convert("RGB"))
        elif f == "auto":
            img = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
        elif f in ("up2", "up3"):
            k = int(f[2])
            img = img.resize((img.width * k, img.height * k), Image.LANCZOS)
        elif f == "sharp":
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
        elif f == "gray":
            img = img.convert("L").convert("RGB")
        elif f == "minch":  # min of R,G,B: red and black backgrounds both go dark, white digits stay bright
            img = Image.fromarray(np.asarray(img.convert("RGB")).min(axis=2)).convert("RGB")
        else:
            raise ValueError(f)
    return img


def png64(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ollama(model, prompt, img):
    body = {"model": model, "stream": False, "keep_alive": common.KEEP_ALIVE, "logprobs": True, "top_logprobs": 3,
            "options": {"temperature": 0, "num_predict": 60},
            "messages": [{"role": "user", "content": prompt, "images": [png64(img)]}]}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        res = json.loads(r.read())
    digits = []
    for t in res.get("logprobs") or []:
        p = math.exp(t["logprob"])
        alt = [(a["token"], round(math.exp(a["logprob"]), 4)) for a in t.get("top_logprobs", [])]
        digits += [{"d": c, "p": round(p, 4), "alt": alt} for c in t["token"] if c in "0123456789"]
    return res["message"]["content"], digits


def tesseract(img):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.save(f.name)
        res = subprocess.run(["tesseract", f.name, "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789",
                              "tsv"], capture_output=True, text=True)
    if res.returncode:  # it crashes (SIGFPE) on some crops: no digits, error kept
        return f"<tesseract exit {res.returncode}>", []
    tsv = res.stdout
    text, digits = [], []
    for line in tsv.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) == 12 and cols[11].strip():
            text.append(cols[11])
            digits += [{"d": c, "p": max(float(cols[10]), 0) / 100} for c in cols[11] if c in "0123456789"]
    return " ".join(text), digits


def ocr(model, filters, part, img):
    """img: a path or a PIL image."""
    img = apply((Image.open(img) if isinstance(img, str) else img).convert("RGB"), filters)
    t0 = time.time()
    text, digits = tesseract(img) if model == "tesseract" else ollama(model, PROMPTS[model][part], img)
    return {"text": text, "digits": digits, "seconds": round(time.time() - t0, 2)}


def widen(model, filters, d, part, count, variants):
    """Read <part>_0, _1, ... until one gives exactly `count` digits; the last try is kept otherwise."""
    tries, res = [], None
    for i in range(variants):
        path = os.path.join(d, f"{part}_{i}.png")
        if not os.path.exists(path):
            break
        res = ocr(model, filters, part, path)
        tries.append({"variant": i, "text": res["text"], "n": len(res["digits"]), "s": res["seconds"]})
        if len(res["digits"]) == count:
            break
    if res is not None:
        res.update(variant=tries[-1]["variant"], tries=tries, seconds=round(sum(t["s"] for t in tries), 2))
    return res


def read_parts(model, filters, name):
    """Black (5) and red (3) wheels read apart; a part with the wrong count becomes all '?'."""
    out = {"file": name}
    d = os.path.join(CACHE, "crops", name)
    black = widen(model, filters, d, "black", 5, len(PART_PADS))
    red = widen(model, filters, d, "red", 3, len(PART_PADS))
    if black is None or red is None:
        return out
    unknown = lambda n: [{"d": "?", "p": 0.0}] * n
    out["black"], out["red"] = black, red
    out["counter"] = {"text": f"{black['text']} | {red['text']}",
                      "digits": (black["digits"] if len(black["digits"]) == 5 else unknown(5))
                                + (red["digits"] if len(red["digits"]) == 3 else unknown(3)),
                      "seconds": round(black["seconds"] + red["seconds"], 2)}
    if os.path.exists(os.path.join(d, "serial.png")):
        out["serial"] = ocr(model, filters, "serial", os.path.join(d, "serial.png"))
    return out


def read_widening(model, filters, part, images, count):
    """OCR images (widening crops of one part) in turn until one gives exactly `count` digits; else the last."""
    res, tries = None, []
    for i, img in enumerate(images):
        res = ocr(model, filters, part, img)
        tries.append({"variant": i, "text": res["text"], "n": len(res["digits"]), "s": res["seconds"]})
        if len(res["digits"]) == count:
            break
    if res is not None:
        res.update(variant=tries[-1]["variant"], tries=tries, seconds=round(sum(t["s"] for t in tries), 2))
    return res


def read(model, filters, name):
    """Counter: widening crops in turn until exactly 8 digits (the wheel count); the last try is kept otherwise."""
    out = {"file": name}
    d = os.path.join(CACHE, "crops", name)
    tries = []
    for i in range(len(PADS)):
        path = os.path.join(d, f"counter_{i}.png")
        if not os.path.exists(path):
            break
        r = ocr(model, filters, "counter", path)
        tries.append({"variant": i, "text": r["text"], "n": len(r["digits"])})
        out["counter"] = {**r, "variant": i, "tries": tries, "seconds": round(sum(t.get("s", 0) for t in tries) + r["seconds"], 2)}
        tries[-1]["s"] = r["seconds"]
        if len(r["digits"]) == 8:
            break
    if os.path.exists(os.path.join(d, "serial.png")):
        out["serial"] = ocr(model, filters, "serial", os.path.join(d, "serial.png"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="glm-ocr")
    p.add_argument("--filter", default="none")
    p.add_argument("--force", action="store_true")
    p.add_argument("--parts", action="store_true", help="read black and red wheels apart")
    p.add_argument("names", nargs="*")
    a = p.parse_args()
    names = a.names or sorted(os.listdir(os.path.join(CACHE, "crops")))
    run = f"{a.model.replace(':', '_')}__{a.filter}" + ("__parts" if a.parts else "")
    for n in names:
        js = cache_path("read", run, n + ".json")
        if os.path.exists(js) and not a.force:
            continue
        r = (read_parts if a.parts else read)(a.model, a.filter, n)
        with open(js, "w") as f:
            json.dump(r, f, ensure_ascii=False)
    print(run, file=sys.stderr)


if __name__ == "__main__":
    main()
