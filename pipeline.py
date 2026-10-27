"""Read a batch of water-meter photos in memory: what the meter-ocr service runs.

Phases keep model swaps down (Ollama holds one model at a time here): Qwen3-VL locates every photo and reads
every counter, then glm-ocr reads every counter and serial. A wheel keeps its digit only where both models
agree; a reading that isn't 8 digits or whose first digit isn't 0 leaves all black wheels '?'.
Result per photo: {"file", "serial", "black" (5 chars), "red" (3 chars), "note"}, '?' for unread wheels;
the HA script matches the serial and votes over photos.
"""
import cv2
import numpy as np
from PIL import Image
from combine import wheels
from crop import PADS, crops
from locate import locate
from read import read_widening

READERS = ("qwen3-vl:8b-instruct", "glm-ocr")


def recognize(photos, log=print):
    """photos: [(name, path)]. Returns one result dict per photo, in order."""
    items = []
    for name, path in photos:
        item = {"file": name, "reads": {}}
        try:
            loc, dial = locate(path)
            item["crops"] = crops(cv2.cvtColor(dial, cv2.COLOR_RGB2BGR), loc) if dial is not None else None
            if item["crops"] is None:
                item["note"] = "не знайдено вікно з колесами"
        except Exception as e:  # one bad photo must not sink the batch
            item["crops"], item["note"] = None, f"помилка пошуку: {e!r}"
        items.append(item)
        log(f"located {name}: {item.get('note', 'ok')}")
    for model in READERS:  # Qwen is still loaded after locating
        for item in items:
            c = item["crops"]
            if c is None:
                continue
            pil = lambda k: Image.fromarray(cv2.cvtColor(c[k], cv2.COLOR_BGR2RGB))
            counter = [pil(f"counter_{i}") for i in range(len(PADS))]
            item["reads"][model] = {"counter": read_widening(model, "none", "counter", counter, 8)}
            if model == "glm-ocr" and "serial" in c:
                item["reads"][model]["serial"] = read_widening(model, "none", "serial", [pil("serial")], 0)
        log(f"read with {model}")
    out = []
    for item in items:
        res = {"file": item["file"], "serial": "", "black": "?????", "red": "???", "note": item.get("note", "")}
        if item["crops"] is not None:
            a, b = (wheels(item["reads"][m]) for m in READERS)
            digits = "".join(x["d"] if x["d"] == y["d"] else "?" for x, y in zip(a, b))
            res["black"], res["red"] = digits[:5], digits[5:]
            serial = item["reads"]["glm-ocr"].get("serial")
            res["serial"] = "".join(d["d"] for d in serial["digits"]) if serial else ""
            if not res["serial"]:
                res["note"] = "не видно серійного номера"
            res["models"] = {m: item["reads"][m]["counter"]["text"] for m in READERS}
        out.append(res)
    return out
