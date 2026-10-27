"""Combine two read runs into one: a wheel keeps its digit only where both runs read the same digit there.

The whole-row and the black/red-parts reads come from different crops, so their mistakes rarely coincide.
A run whose black wheels don't start with "0" counts as unread (household meters are far below 10 000 m³;
a 5-wheel reading like 01008 is fine).
Writes <cache>/read/<a>+<b>/<name>.json in read.py's format (serial taken from run a).
"""
import argparse, json, os
from common import CACHE, cache_path


def wheels(r):
    digits = [d for d in (r.get("counter") or {}).get("digits", []) if d["d"] in "0123456789?"]
    if len(digits) != 8 or digits[0]["d"] != "0":
        return [{"d": "?", "p": 0.0}] * 8
    return digits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    args = p.parse_args()
    run = f"{args.a}+{args.b}"
    for f in sorted(os.listdir(os.path.join(CACHE, "read", args.a))):
        pb = os.path.join(CACHE, "read", args.b, f)
        if not os.path.exists(pb):
            continue
        with open(os.path.join(CACHE, "read", args.a, f)) as fa, open(pb) as fb:
            ra, rb = json.load(fa), json.load(fb)
        digits = [x if x["d"] == y["d"] and x["d"] != "?" else {"d": "?", "p": 0.0}
                  for x, y in zip(wheels(ra), wheels(rb))]
        out = {"file": ra["file"], "counter": {"text": "", "digits": [{**d, "p": min(d["p"], 1.0)} for d in digits],
                                                "seconds": ra["counter"]["seconds"] + rb["counter"]["seconds"]}}
        if ra.get("serial"):
            out["serial"] = ra["serial"]
        with open(cache_path("read", run, f), "w") as fh:
            json.dump(out, fh, ensure_ascii=False)
    print(run)


if __name__ == "__main__":
    main()
