#!/usr/bin/env python3
"""bin_recall.py v2 — coverage, windows, pre-filter recall and 3 s-grid frame
recall for B in {50,100,200,500} (median + 2*MAD)."""
import argparse, os, zipfile, statistics
import numpy as np

FPS = 30.0
DELTA = 3.0


def load_gt(path):
    gt = {}
    for line in open(path, encoding="utf-8", errors="ignore"):
        p = line.split()
        if len(p) < 6 or p[1].lower().startswith("normal"):
            continue
        name = p[0][:-4] if p[0].lower().endswith(".mp4") else p[0]
        iv = []
        for a, b in ((p[2], p[3]), (p[4], p[5])):
            try:
                a, b = int(a), int(b)
            except ValueError:
                continue
            if a >= 0 and b > a:
                iv.append((a / FPS, b / FPS))
        if iv:
            gt[name] = (p[1], iv)
    return gt


def find_npz(base, cls, name):
    for c in (cls, cls.replace("RoadAccidents", "Road_Accidents"), cls.lower()):
        p = os.path.join(base, c, name + ".npz")
        if os.path.exists(p) and zipfile.is_zipfile(p):
            return p
    return None


def windows(C, edges):
    med = np.median(C)
    mad = np.median(np.abs(C - med))
    act = np.flatnonzero(C > med + 2 * mad)
    out = []
    if act.size == 0:
        return out
    s = p = act[0]
    for b in act[1:]:
        if b == p + 1:
            p = b
            continue
        out.append((edges[s], edges[p + 1]))
        s = p = b
    out.append((edges[s], edges[p + 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--bins", default="50,100,200,500")
    a = ap.parse_args()
    Bs = [int(x) for x in a.bins.split(",")]

    gt = load_gt(a.gt)
    res = {B: {"cov": [], "nw": [], "hit": 0, "an": 0.0, "cs": 0.0,
               "fhit": 0, "nf": [], "short": [0, 0]} for B in Bs}
    import time
    n = 0
    items = sorted(gt.items())
    t0 = time.time()
    for i, (name, (cls, iv)) in enumerate(items, 1):
        if i % 20 == 0 or i == len(items):
            el = time.time() - t0
            print(f"  [{i}/{len(items)}] used={n}  {el:.0f}s elapsed, "
                  f"~{el / i * (len(items) - i):.0f}s left", flush=True)
        p = find_npz(a.npz, cls, name)
        if p is None:
            continue
        try:
            t = np.load(p)["t"] / 1e6
        except Exception:
            continue
        if t.size == 0 or t.max() <= t.min():
            continue
        t = t - t.min()
        T = float(t.max())
        if max(e for _, e in iv) > T * 1.10:
            continue
        n += 1
        grid = np.arange(0.0, T + 1e-9, DELTA)
        for B in Bs:
            C, edges = np.histogram(t, bins=B)
            W = windows(C.astype(float), edges)
            r = res[B]
            r["cov"].append(100.0 * sum(e - s for s, e in W) / T)
            r["nw"].append(len(W))
            r["hit"] += any(min(a1, w1) > max(a0, w0) for a0, a1 in iv for w0, w1 in W)
            r["an"] += sum(e - s for s, e in iv)
            r["cs"] += sum(max(0.0, min(a1, w1) - max(a0, w0))
                           for a0, a1 in iv for w0, w1 in W)
            F = [g for g in grid if any(w0 <= g <= w1 for w0, w1 in W)]
            r["nf"].append(len(F))
            r["fhit"] += any(a0 <= g <= a1 for g in F for a0, a1 in iv)
            r["short"][0] += sum(1 for w0, w1 in W
                                 if not any(w0 <= g <= w1 for g in grid))
            r["short"][1] += len(W)

    print(f"usable videos: {n}")
    print(f"{'B':>5}{'Cov %':>8}{'Win/vid':>9}{'Recall %':>10}{'AnomTime %':>12}"
          f"{'Frames/vid':>12}{'FrameRecall %':>15}{'Empty win %':>13}")
    for B in Bs:
        r = res[B]
        print(f"{B:>5}{statistics.mean(r['cov']):>8.1f}{statistics.mean(r['nw']):>9.1f}"
              f"{100.0 * r['hit'] / n:>10.1f}{100.0 * r['cs'] / r['an']:>12.1f}"
              f"{statistics.mean(r['nf']):>12.1f}{100.0 * r['fhit'] / n:>15.1f}"
              f"{100.0 * r['short'][0] / max(r['short'][1], 1):>13.1f}")
    print("\nFrameRecall = a scheduled 3 s frame lands inside the annotated anomaly."
          "\nEmpty win   = selected windows that contain no scheduled frame at all.")


if __name__ == "__main__":
    main()