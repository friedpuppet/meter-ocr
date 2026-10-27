"""Find the counter window and the serial number on a water-meter photo; read nothing.

Stage 1: Qwen3-VL boxes the dial, the counter window, the NOVATOR logo and the serial on the whole photo.
Coarse angle: NOVATOR is always above the dial's centre and the serial below it, so they give "up".
Stage 2: Qwen3-VL boxes the counter window and the serial on the coarsely levelled dial.
Fine angle (code): the counter frame's long straight edges (Hough) inside that box.
Counter box (code): each side of Qwen's box snapped onto the frame's outer edge.
Result per photo: <cache>/locate/<name>.json, plus the levelled dial as <cache>/dial/<name>.jpg.
"""
import argparse, json, math, os, sys
import cv2
import numpy as np
from common import LOCATOR, b64, cache_path, chat, load_rgb

BOX = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
NBOX = {"anyOf": [BOX, {"type": "null"}]}
STAGE1 = """This photo shows a household water meter (round dial under glass). Locate, do not read:
- "dial": bounding box of the whole round dial face.
- "counter": bounding box of the digit counter window: the row of number wheels (5 black, then 3 red) in a
  red or blue frame.
- "logo": bounding box of the printed word "NOVATOR", or null if it is not visible.
- "serial": bounding box of the serial number printed after "№" on the dial, or null if it is not visible.
Boxes are [x1, y1, x2, y2] in coordinates relative 0-1000."""
STAGE2 = """This is a water meter dial, rotated so the digit counter is horizontal. Locate, do not read:
- "counter": bounding box of the digit counter window with its frame: all 8 number wheels (5 black, then 3 red).
- "serial": bounding box of the whole serial number printed after "№", all its digits, or null if it is not
  visible.
Boxes are [x1, y1, x2, y2] in coordinates relative 0-1000."""
SCHEMA1 = {"type": "object", "required": ["dial", "counter", "logo", "serial"],
           "properties": {"dial": BOX, "counter": BOX, "logo": NBOX, "serial": NBOX}}
SCHEMA2 = {"type": "object", "required": ["counter", "serial"], "properties": {"counter": BOX, "serial": NBOX}}
DIAL_SIDE = 1.4  # levelled dial crop: this many times the dial box, so our crop never cuts the meter


def to_px(box, w, h):
    if not box:
        return None
    x1, y1, x2, y2 = box
    return [min(x1, x2) * w / 1000, min(y1, y2) * h / 1000, max(x1, x2) * w / 1000, max(y1, y2) * h / 1000]


def ask(prompt, schema, img):
    text, sec = chat(LOCATOR, prompt, [b64(img, 1024)], fmt=schema)
    return json.loads(text), sec


def red_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60))


def centre(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def coarse_angle(dial, counter, logo, serial):
    """Rotation (degrees, cv2 sense) that puts "up" (dial centre -> logo, serial -> dial centre) at the top."""
    c = centre(dial)
    ups = [v / np.linalg.norm(v) for v in (centre(logo) - c if logo else None, c - centre(serial) if serial else None)
           if v is not None and np.linalg.norm(v) > 1]
    if not ups:
        return None
    u = sum(ups)
    return math.degrees(math.atan2(u[1], u[0])) + 90


WORK = 1200  # the dial is scaled to this width for line finding


def dial_lines(img, dial, max_angle):
    """Long straight edges inside the dial's circle: [(angle, length, x1, y1, x2, y2)] in img coordinates.

    Angles in cv2 sense (positive = clockwise in the image), only those within max_angle of horizontal. The
    counter frame gives the longest such lines; the dial's rim is round and the rotation's canvas edge lies
    outside the circle.
    """
    sc = WORK / img.shape[1]
    small = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0), 30, 90)
    x1, y1, x2, y2 = [v * sc for v in dial]
    inside = np.zeros_like(edges)
    cv2.ellipse(inside, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (int(0.46 * (x2 - x1)), int(0.46 * (y2 - y1))),
                0, 0, 360, 255, -1)
    edges &= inside
    dw = x2 - x1
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 40, minLineLength=int(0.25 * dw), maxLineGap=int(0.03 * dw))
    out = []
    for lx1, ly1, lx2, ly2 in (lines.reshape(-1, 4) if lines is not None else []):
        a = (math.degrees(math.atan2(ly2 - ly1, lx2 - lx1)) + 90) % 180 - 90
        if abs(a) <= max_angle:
            out.append((a, math.hypot(lx2 - lx1, ly2 - ly1) / sc, lx1 / sc, ly1 / sc, lx2 / sc, ly2 / sc))
    return out


def frame_angle(lines):
    """Peak of the angle histogram, weighted by length squared so the frame's long edges dominate."""
    if not lines:
        return 0.0
    bins = np.arange(-40, 40.5, 0.5)
    hist = np.zeros(len(bins))
    for a, l, *_ in lines:
        hist[np.argmin(np.abs(bins - a))] += l * l
    hist = np.convolve(hist, np.ones(5), "same")
    return float(bins[np.argmax(hist)])


FRAME_ASPECT = 4.5  # outer frame of the NOVATOR counter window, width : height


def frame_box(lines, hint, rgb):
    """Counter frame on a level image from its horizontal edges inside Qwen's box (hint).

    Ends: the long lines' end points when they are within 15 % of the hint's ends (the frame's edge may be lost
    in glare or dark), else the hint's. Top/bottom: the pair of lines whose distance fits the frame's known
    aspect (+-20 %) and overlap the hint's rows by half or more, best by: overlap with the hint's rows x the lines' length / brightness of the band between
    them (the black wheels sit there); with no such pair, that height around the hint's middle.
    None when no line fits at all.
    """
    x1, y1, x2, y2 = hint
    bw, bh = x2 - x1, y2 - y1

    def inside_share(l):
        a, b = min(l[2], l[4]), max(l[2], l[4])
        return max(0.0, min(b, x2) - max(a, x1)) / max(b - a, 1)

    near = [l for l in lines if abs(l[0]) < 2 and y1 - 0.6 * bh <= (l[3] + l[5]) / 2 <= y2 + 0.6 * bh
            and inside_share(l) > 0.5]
    if not near:
        return None, None
    longest = max(l[1] for l in near)
    long_ = [l for l in near if l[1] >= 0.5 * longest]
    left = min(min(l[2], l[4]) for l in long_)
    right = max(max(l[2], l[4]) for l in long_)
    left = left if abs(left - x1) <= 0.15 * bw else x1
    right = right if abs(right - x2) <= 0.15 * bw else x2
    want = (right - left) / FRAME_ASPECT
    ys = sorted(((l[3] + l[5]) / 2, l[1]) for l in near)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bx1, bx2 = int(max(left, 0)), int(left + 0.6 * (right - left))  # the black wheels' part
    best = None
    for i, (ya, la) in enumerate(ys):
        for yb, lb in ys[i + 1:]:
            if not 0.8 * want <= yb - ya <= 1.2 * want:
                continue
            band = gray[int(ya + 0.2 * (yb - ya)):int(yb - 0.2 * (yb - ya)), bx1:bx2]
            iou = max(0.0, min(yb, y2) - max(ya, y1)) / (max(yb, y2) - min(ya, y1))
            if iou < 0.5:  # e.g. an edge above the window + one inside it, when the real edge is lost in glare
                continue
            score = iou * (la + lb) / (float(band.mean()) + 20) if band.size else 0.0
            if best is None or score > best[0]:
                best = (score, ya, yb)
    if best:
        return [left, best[1], right, best[2]], "frame"
    mid = (y1 + y2) / 2
    return [left, mid - want / 2, right, mid + want / 2], "frame-ends"


def sharpness(rgb, box):
    """Variance of the Laplacian of the counter crop scaled to 100 px high: low = out of focus."""
    x1, y1, x2, y2 = map(int, box)
    g = cv2.cvtColor(rgb[max(y1, 0):y2, max(x1, 0):x2], cv2.COLOR_RGB2GRAY)
    if g.size == 0:
        return 0.0
    g = cv2.resize(g, (max(1, int(g.shape[1] * 100 / g.shape[0])), 100), interpolation=cv2.INTER_AREA)
    return round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1)


def level(rgb, angle, center):
    """Rotate the whole photo about center by angle; canvas grows to fit. Returns image and 2x3 matrix."""
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D(tuple(map(float, center)), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - center[0]
    m[1, 2] += nh / 2 - center[1]
    return cv2.warpAffine(rgb, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=(0, 0, 0)), m


def map_box(m, box):
    """Axis-aligned box around the rotated corners of box."""
    x1, y1, x2, y2 = box
    pts = m @ np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], float).T
    return [pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()]


def map_dial(m, box):
    """The dial is round: after a rotation it keeps its size, only its centre moves."""
    cx, cy = m @ np.array([*centre(box), 1.0])
    hw, hh = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
    return [cx - hw, cy - hh, cx + hw, cy + hh]


def square(img, centre_xy, side):
    x1, y1 = int(max(centre_xy[0] - side / 2, 0)), int(max(centre_xy[1] - side / 2, 0))
    x2, y2 = int(min(centre_xy[0] + side / 2, img.shape[1])), int(min(centre_xy[1] + side / 2, img.shape[0]))
    return img[y1:y2, x1:x2], (x1, y1)


def shift(box, dx, dy):
    return [box[0] - dx, box[1] - dy, box[2] - dx, box[3] - dy] if box else None


def locate(path, stage1=None, stage2=None):
    rgb = load_rgb(path)
    h, w = rgb.shape[:2]
    t1 = 0.0
    if stage1 is None:
        stage1, t1 = ask(STAGE1, SCHEMA1, rgb)
    dial, counter = to_px(stage1["dial"], w, h), to_px(stage1["counter"], w, h)
    logo, serial = to_px(stage1["logo"], w, h), to_px(stage1["serial"], w, h)
    out = {"file": os.path.basename(path), "size": [w, h], "stage1": stage1}
    coarse = coarse_angle(dial, counter, logo, serial)
    out["coarse"] = None if coarse is None else round(coarse, 2)
    # Coarse level by the logo/serial, then crop the dial generously.
    rot, m = level(rgb, coarse or 0.0, centre(dial))
    d = map_dial(m, dial)
    crop1, (ox, oy) = square(rot, centre(d), max(d[2] - d[0], d[3] - d[1]) * DIAL_SIDE)
    # Fine level by the frame's straight edges (twice: the second pass starts nearly level).
    dial1 = map_dial(m, dial)
    dial1 = [dial1[0] - ox, dial1[1] - oy, dial1[2] - ox, dial1[3] - oy]
    crop, dialf, fine = crop1, dial1, 0.0
    for max_angle in (40, 8):
        fine += frame_angle(dial_lines(crop, dialf, max_angle))
        crop, m2 = level(crop1, fine, centre(dial1))
        dialf = map_dial(m2, dial1)
    # Qwen boxes the counter and serial on the level dial; the frame's own edges then set the counter box.
    # The levelled dial depends only on stage 1, so the stage-2 answer is reused when the angle is unchanged.
    angle = round((coarse or 0.0) + fine, 2)
    t2 = 0.0
    if stage2 and stage2[1] == angle:
        r2 = stage2[0]
    else:
        r2, t2 = ask(STAGE2, SCHEMA2, crop)
    ch, cw = crop.shape[:2]
    hint, ser = to_px(r2["counter"], cw, ch), to_px(r2["serial"], cw, ch)
    lines = dial_lines(crop, dialf, 2)
    box, source = frame_box(lines, hint, crop)
    out.update(stage2=r2, fine=round(fine, 2), lines=len(lines), angle=angle,
               counter=box or hint, counter_from=source or "qwen", serial=ser,
               sharpness=sharpness(crop, box or hint), seconds=[round(t1, 1), round(t2, 1)])
    return out, crop


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="*", help="photo names in the photo dir (default: all)")
    p.add_argument("--dir", required=True, help="directory with the photos")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    files = a.files or sorted(f for f in os.listdir(a.dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    for f in files:
        name = os.path.splitext(f)[0]
        js = cache_path("locate", name + ".json")
        old = old2 = None
        if os.path.exists(js):
            if not a.force:
                continue
            with open(js) as fh:
                prev = json.load(fh)
            old = prev.get("stage1")
            old2 = (prev["stage2"], prev.get("angle")) if prev.get("stage2") else None
        try:  # the whole-photo answer is reused when it has every field the current prompt asks for
            out, crop = locate(os.path.join(a.dir, f), old if old and set(old) >= set(SCHEMA1["required"]) else None, old2)
        except Exception as e:  # keep going; the failure is recorded
            out, crop = {"file": f, "error": repr(e)}, None
        if crop is not None:
            cv2.imwrite(cache_path("dial", name + ".jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(js, "w") as fh:
            json.dump(out, fh, ensure_ascii=False)
        print(f, out.get("coarse"), out.get("fine"), out.get("counter_from"), out.get("lines"), out.get("sharpness"), out.get("error", ""), file=sys.stderr)


if __name__ == "__main__":
    main()
