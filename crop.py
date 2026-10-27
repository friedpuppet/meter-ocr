"""Cut the counter window and the serial number out of the levelled dials; read nothing.

Input: <cache>/locate/<name>.json + <cache>/dial/<name>.jpg (from locate.py).
Output: <cache>/crops/<name>/{counter,counter_0..3,serial}.png
"""
import argparse, json, os
import cv2
import numpy as np
from common import CACHE

PADS = [0.03, 0.10, 0.18, 0.28]  # horizontal padding of the counter crop, share of its width
SPLIT = 0.665  # usual black/red boundary, share of the frame's width
PART_PADS = [0.03, 0.08, 0.15]  # outer padding of the black / red part, share of the whole window's width


def cut(img, box, pad_x=0.0, pad_y=0.0):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad_x, (y2 - y1) * pad_y
    return img[int(max(y1 - py, 0)):int(min(y2 + py, h)), int(max(x1 - px, 0)):int(min(x2 + px, w))]


def shear_of(img):
    """Horizontal shear (dx per dy) that makes the window's vertical structure upright: frame sides, wheel
    borders, digit strokes. Picked by the peakiest column profile of vertical edges over -30..30 degrees."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 1, 0))
    h, w = gx.shape
    best, best_k = -1.0, 0.0
    for k in np.tan(np.radians(np.arange(-30, 30.5, 1.0))):
        m = np.float32([[1, k, -k * h / 2], [0, 1, 0]])
        prof = cv2.warpAffine(gx, m, (w, h)).sum(axis=0)
        score = float((prof ** 2).sum())
        if score > best:
            best, best_k = score, float(k)
    return best_k


def red_split(img, box):
    """Column (image x) between the black and the red wheels: the split that best separates a dark-ish left
    from a red right, by redness R - max(G, B) along the window's middle rows, searched at 45-80 % of the width."""
    x1, y1, x2, y2 = map(int, box)
    band = img[int(y1 + 0.35 * (y2 - y1)):int(y1 + 0.65 * (y2 - y1)), x1:x2].astype(np.float32)
    if band.size == 0:
        return None
    red = band[..., 2] - np.maximum(band[..., 0], band[..., 1])  # BGR
    col = np.median(red, axis=0)
    w = len(col)
    best, at = None, None
    for c in range(int(0.45 * w), int(0.80 * w)):
        gap = col[c:].mean() - col[:c].mean()
        if best is None or gap > best:
            best, at = gap, c
    return (x1 + at, float(best)) if at is not None else None


def crops(dial, loc):
    """All crops of one levelled dial (BGR) as {name: image}, plus "meta"; None without a counter box.

    counter_0..3: the window with widening padding, read in order until one gives exactly 8 digits.
    black_*/red_*: the two parts apart, each widening outwards only. serial: the serial number.
    """
    if not loc.get("counter"):
        return None
    out = {}
    # Shear is measured on the window itself and applied about its middle row to the dial.
    x1, y1, x2, y2 = loc["counter"]
    k = shear_of(cut(dial, loc["counter"], 0.0, 0.0))
    yc = (y1 + y2) / 2
    sheared = cv2.warpAffine(dial, np.float32([[1, k, -k * yc], [0, 1, 0]]), (dial.shape[1], dial.shape[0]),
                             flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0))
    for i, pad in enumerate(PADS):
        out[f"counter_{i}"] = cut(sheared, loc["counter"], pad, 0.12)
    split = red_split(sheared, loc["counter"])
    if not split or split[1] < 20 or not 0.58 <= (split[0] - x1) / (x2 - x1) <= 0.76:
        split = (x1 + SPLIT * (x2 - x1), split[1] if split else 0.0)  # red barely visible: the usual place
    bw = x2 - x1
    for i, pad in enumerate(PART_PADS):
        out[f"black_{i}"] = cut(sheared, [x1 - pad * bw, y1, split[0] + 0.01 * bw, y2], 0, 0.12)
        out[f"red_{i}"] = cut(sheared, [split[0] - 0.01 * bw, y1, x2 + pad * bw, y2], 0, 0.12)
    # The serial lies below the window, where the shear would shift it: cut it from the unsheared dial.
    if loc.get("serial"):
        out["serial"] = cut(dial, loc["serial"], 0.15, 0.4)
    out["meta"] = {"shear_deg": round(float(np.degrees(np.arctan(k))), 1),
                   "split": round((split[0] - x1) / (x2 - x1), 3), "split_contrast": round(split[1], 1)}
    return out


def crop(name):
    with open(os.path.join(CACHE, "locate", name + ".json")) as f:
        loc = json.load(f)
    c = crops(cv2.imread(os.path.join(CACHE, "dial", name + ".jpg")), loc)
    if c is None:
        return False
    out = os.path.join(CACHE, "crops", name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "shear.json"), "w") as f:
        json.dump(c.pop("meta"), f)
    c["counter"] = c["counter_0"]
    serial = os.path.join(out, "serial.png")
    if "serial" not in c and os.path.exists(serial):
        os.remove(serial)
    for k, img in c.items():
        cv2.imwrite(os.path.join(out, k + ".png"), img)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    a = p.parse_args()
    names = a.names or sorted(f[:-5] for f in os.listdir(os.path.join(CACHE, "locate")))
    done = sum(crop(n) for n in names)
    print(f"cropped {done} of {len(names)}")


if __name__ == "__main__":
    main()
