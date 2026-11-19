#!/usr/bin/env python3
"""
variant_recall.py — pre-filter recall PER THRESHOLD VARIANT.  R5-#3, and it
repairs the untested prediction sitting in §5.6.

    python evaluation/variant_recall.py \
        --json outputs/windows/global_active_windows.json \
        --gt   data/Temporal_Anomaly_Annotation.txt \
        --npz  data/UCF-Crime-DVS

Recomputes the gate from the event streams under four thresholds

    MAD       med(C) + 2*MAD(C)      <- the paper's
    Mean      mean(C) + 1*std(C)
    Fixed     80th percentile of C
    Median    med(C)

and for each one reports, ON THE SAME SET OF VIDEOS:

    coverage        selected fraction of the timeline
    recall          videos where >=1 window overlaps the annotated anomaly
    VLM calls       mean scheduled frames per video, at 3 s sampling

That last column is R5-#6's "VLM invocation count" for free.

Windows are built exactly as the paper specifies: B = 200 equal bins, bins
active under STRICT inequality c_b > theta, adjacent active bins merged with
no gap tolerance.

SANITY CHECK the script prints: the recomputed MAD coverage should land near
20.7 % and the Fixed/Median variants near 20 % / 50 % by construction. If MAD
is far off, the NPZ subset is not representative and you should say so rather
than publish the table.
"""
import argparse, collections, json, os, statistics, sys, zipfile
import numpy as np

FPS = 30.0
B = 200
DELTA = 3.0          # frame sampling interval, seconds
ORDER = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
         "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
         "Stealing", "Vandalism"]


# ── ground truth ────────────────────────────────────────────────────────
def load_gt(path):
    gt = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            name = p[0][:-4] if p[0].lower().endswith(".mp4") else p[0]
            if p[1].lower().startswith("normal"):
                continue
            iv = []
            for a, b in ((p[2], p[3]), (p[4], p[5])):
                try:
                    a, b = int(a), int(b)
                except ValueError:
                    continue
                if a >= 0 and b > a:
                    iv.append((a / FPS, b / FPS))
            if iv:
                gt[name] = iv
    return gt


def merge(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for s, e in iv[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def overlaps(anom, wins):
    for a0, a1 in anom:
        for w0, w1 in wins:
            if min(a1, w1) > max(a0, w0):
                return True
    return False


def covered_seconds(anom, wins):
    tot = 0.0
    for a0, a1 in anom:
        for w0, w1 in wins:
            lo, hi = max(a0, w0), min(a1, w1)
            if hi > lo:
                tot += hi - lo
    return tot


# ── the gate ────────────────────────────────────────────────────────────
def thresholds(C):
    med = float(np.median(C))
    mad = float(np.median(np.abs(C - med)))
    return {
        "MAD":    med + 2.0 * mad,
        "Mean":   float(C.mean() + C.std()),
        "Fixed":  float(np.percentile(C, 80)),
        "Median": med,
    }


def windows_from(C, thr, edges):
    """Strict inequality, adjacent active bins merged, no gap tolerance."""
    active = np.flatnonzero(C > thr)
    if active.size == 0:
        return []
    out, start, prev = [], active[0], active[0]
    for b in active[1:]:
        if b == prev + 1:
            prev = b
            continue
        out.append((edges[start], edges[prev + 1]))
        start = prev = b
    out.append((edges[start], edges[prev + 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("outputs", "windows",
                                                   "global_active_windows.json"))
    ap.add_argument("--gt", required=True)
    ap.add_argument("--npz", required=True, help="base dir of the NPZ files")
    ap.add_argument("--tol", type=float, default=1.10)
    a = ap.parse_args()

    for p in (a.json, a.gt):
        if not os.path.exists(p):
            sys.exit(f"NOT FOUND: {p}")

    data = json.load(open(a.json, encoding="utf-8"))
    gt = load_gt(a.gt)
    print(f"annotations : {len(gt)} anomalous videos")
    print(f"records     : {len(data)} in the JSON\n")

    VARIANTS = ["MAD", "Mean", "Fixed", "Median"]
    agg = {v: {"cov": [], "hit": 0, "anom": 0.0, "cov_s": 0.0, "frames": []}
           for v in VARIANTS}
    per_class = {v: collections.defaultdict(lambda: [0, 0]) for v in VARIANTS}

    used = skipped_nonpz = skipped_nogt = skipped_align = 0

    for r in data:
        name = r.get("video_name", "")
        if name not in gt:
            skipped_nogt += 1
            continue
        cls = r.get("class_name", "?")
        path = os.path.join(a.npz, cls, name + ".npz")
        if not (os.path.exists(path) and zipfile.is_zipfile(path)):
            skipped_nonpz += 1
            continue
        try:
            t = np.load(path)["t"] / 1e6
        except Exception:
            skipped_nonpz += 1
            continue
        if t.size == 0 or t.max() <= t.min():
            skipped_nonpz += 1
            continue

        T = float(t.max() - t.min())
        anom = merge(gt[name])
        if max(e for _, e in anom) > T * a.tol:
            skipped_align += 1
            continue

        C, edges = np.histogram(t, bins=B)
        edges = edges - edges[0]                 # windows in stream time
        thr = thresholds(C.astype(float))
        anom0 = [(s, e) for s, e in anom]
        anom_total = sum(e - s for s, e in anom0)
        used += 1

        for v in VARIANTS:
            W = windows_from(C.astype(float), thr[v], edges)
            sel = sum(w1 - w0 for w0, w1 in W)
            agg[v]["cov"].append(100.0 * sel / T if T > 0 else 0.0)
            agg[v]["frames"].append(sel / DELTA)
            hit = overlaps(anom0, W)
            agg[v]["hit"] += 1 if hit else 0
            agg[v]["anom"] += anom_total
            agg[v]["cov_s"] += covered_seconds(anom0, W)
            pc = per_class[v][cls]
            pc[0] += 1
            pc[1] += 1 if hit else 0

    print("=" * 72)
    print(f"usable videos            : {used}")
    print(f"  skipped, no annotation : {skipped_nogt}")
    print(f"  skipped, no/bad NPZ    : {skipped_nonpz}")
    print(f"  skipped, misaligned    : {skipped_align}")
    print("=" * 72)
    if used == 0:
        sys.exit("nothing to report")
    if used < 60:
        print("\n!! Fewer than 60 videos. Per-class numbers will be noise;"
              "\n   report the overall row only, or say the denominator.\n")

    print(f"\n{'Variant':<10}{'Threshold':<22}{'Cov %':>8}{'Recall %':>10}"
          f"{'DurRec %':>10}{'Frames/vid':>12}")
    print("-" * 72)
    rows = []
    for v, label in (("MAD", "med + 2*MAD"), ("Mean", "mean + 1*sigma"),
                     ("Fixed", "80th percentile"), ("Median", "median only")):
        d = agg[v]
        cov = statistics.mean(d["cov"])
        rec = 100.0 * d["hit"] / used
        dur = 100.0 * d["cov_s"] / d["anom"] if d["anom"] > 0 else float("nan")
        frm = statistics.mean(d["frames"])
        rows.append((v, label, cov, rec, dur, frm))
        print(f"{v:<10}{label:<22}{cov:>8.1f}{rec:>10.1f}{dur:>10.1f}{frm:>12.1f}")

    mad_cov = rows[0][2]
    print("\nSANITY: recomputed MAD coverage = "
          f"{mad_cov:.1f} % (paper reports 20.7 % over all 969). "
          f"{'OK' if abs(mad_cov-20.7) < 4 else 'CHECK THIS BEFORE PUBLISHING'}")
    print(f"        Fixed should be ~20 %, Median ~50 % by construction.")

    print("\n" + "=" * 72)
    print("LATEX — drop-in rows for the ablation table")
    print("=" * 72)
    for v, label, cov, rec, dur, frm in rows:
        nm = "\\textbf{E-VEZA (ours)}" if v == "MAD" else f"E-VEZA-{v}"
        print(f"{nm:<24}& {label:<18}& {cov:5.1f} & {rec:5.1f} & {frm:5.1f} \\\\")

    print("\n" + "=" * 72)
    print("PER-CLASS RECALL (n per class in brackets)")
    print("=" * 72)
    print(f"{'Class':<16}" + "".join(f"{v:>12}" for v in VARIANTS))
    for c in ORDER:
        if c not in per_class["MAD"]:
            continue
        line = f"{c:<16}"
        for v in VARIANTS:
            n, h = per_class[v][c]
            line += f"{100.0*h/n:>8.1f}({n:>2})"
        print(line)

    print("\nSend me this whole block.")


if __name__ == "__main__":
    main()
