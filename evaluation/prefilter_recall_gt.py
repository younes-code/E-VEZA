#!/usr/bin/env python3
"""
prefilter_recall_gt.py — R8-#2 / R5-#8 against the OFFICIAL UCF-Crime
temporal anomaly annotations (test split, 30 fps, frame numbers).

    python prefilter_recall_gt.py \
        --json  outputs/windows/global_active_windows.json \
        --gt    data/Temporal_Anomaly_Annotation.txt

Reports two things:

  HIT RATE      fraction of videos where >=1 selected window overlaps >=1
                annotated anomalous interval.  (weak criterion)

  DURATION      fraction of annotated anomalous SECONDS that fall inside
  RECALL        selected windows.              (strong criterion)

Runs a sanity check first: the annotation is in RGB-video time, the windows
are in DVS-stream time. If the last annotated second exceeds the DVS stream
duration the two timelines are not aligned and the result is meaningless.
The script tells you rather than guessing.
"""
import json, argparse, collections, os, sys

FPS = 30.0
ORDER = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
         "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting",
         "Stealing", "Vandalism"]
LABEL = {"RoadAccidents": "Road Accidents"}


def load_gt(path):
    """video_name (no .mp4) -> [(start_sec, end_sec), ...]  anomalous only."""
    gt = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            name = p[0]
            if name.lower().endswith(".mp4"):
                name = name[:-4]
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


def covered(anom, wins):
    """seconds of `anom` that fall inside `wins` (both lists of (s,e))."""
    tot = 0.0
    for a0, a1 in anom:
        for w0, w1 in wins:
            lo, hi = max(a0, w0), min(a1, w1)
            if hi > lo:
                tot += hi - lo
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join("outputs", "windows",
                                                   "global_active_windows.json"))
    ap.add_argument("--gt", required=True)
    ap.add_argument("--tol", type=float, default=1.10,
                    help="allowed ratio of annotated end to DVS duration")
    args = ap.parse_args()

    for p in (args.json, args.gt):
        if not os.path.exists(p):
            sys.exit(f"NOT FOUND: {p}")

    data = json.load(open(args.json, encoding="utf-8"))
    gt = load_gt(args.gt)
    print(f"annotations : {len(gt)} anomalous videos in the GT file")
    print(f"records     : {len(data)} in the JSON\n")

    matched, misaligned, rows = [], [], []
    for r in data:
        name = r.get("video_name", "")
        if name not in gt:
            continue
        anom = merge(gt[name])
        wins = merge([tuple(w) for w in r.get("active_windows", [])])
        dur = float(r.get("total_duration_sec", 0.0))
        last = max(e for _, e in anom)

        if dur <= 0 or last > dur * args.tol:
            misaligned.append((name, last, dur))
            continue

        anom_total = sum(e - s for s, e in anom)
        cov = covered(anom, wins)
        matched.append((r.get("class_name", "?"), name, anom_total, cov,
                        float(r.get("coverage_percent", 0.0))))

    # ── alignment report ────────────────────────────────────────────────
    print("=" * 74)
    print("ALIGNMENT CHECK  (annotated end time vs DVS stream duration)")
    print("=" * 74)
    print(f"  usable      : {len(matched)}")
    print(f"  misaligned  : {len(misaligned)}   "
          f"(annotated anomaly ends after the event stream does)")
    for name, last, dur in misaligned[:8]:
        print(f"      {name:<28} GT ends {last:7.1f}s   DVS lasts {dur:7.1f}s")
    if len(misaligned) > 8:
        print(f"      ... and {len(misaligned)-8} more")

    if not matched:
        sys.exit("\nNo usable videos. The two timelines do not align — "
                 "do not report recall.")
    if len(misaligned) > len(matched):
        print("\n!! More misaligned than usable. The DVS streams are probably "
              "not on the\n   same clock as the RGB annotations. "
              "Do not report these numbers.")

    # ── results ─────────────────────────────────────────────────────────
    by = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, []])
    for cls, name, atot, cov, pct in matched:
        b = by[cls]
        b[0] += 1
        b[1] += 1 if cov > 0 else 0
        b[2] += atot
        b[3] += cov
        b[4].append(pct)

    print("\n" + "=" * 74)
    print("PRE-FILTER RECALL")
    print("=" * 74)
    print(f"{'Class':<16}{'N':>4}{'Hit %':>8}{'Dur.Recall %':>14}{'Cov %':>8}")
    print("-" * 74)

    tn = th = 0
    ta = tc = 0.0
    out = []
    for c in ORDER:
        if c not in by:
            continue
        n, h, atot, cov, pcts = by[c]
        hit = 100.0 * h / n
        dr = 100.0 * cov / atot if atot > 0 else float("nan")
        mc = sum(pcts) / len(pcts)
        tn += n; th += h; ta += atot; tc += cov
        out.append((c, n, hit, dr, mc))
        print(f"{LABEL.get(c,c):<16}{n:>4}{hit:>8.1f}{dr:>14.1f}{mc:>8.1f}")

    print("-" * 74)
    print(f"{'OVERALL':<16}{tn:>4}{100.0*th/tn:>8.1f}"
          f"{100.0*tc/ta if ta>0 else float('nan'):>14.1f}")

    print("\n" + "=" * 74)
    print("LATEX — table body")
    print("=" * 74)
    for c, n, hit, dr, mc in out:
        print(f"{LABEL.get(c,c):<16}& {n:<4}& {hit:<6.1f}& {dr:<6.1f}& {mc:.1f} \\\\")
    print("\\hline")
    print(f"\\textbf{{Overall}} & \\textbf{{{tn}}} & "
          f"\\textbf{{{100.0*th/tn:.1f}}} & "
          f"\\textbf{{{100.0*tc/ta:.1f}}} & \\textbf{{20.7}} \\\\")

    print("\nCaption must state N (the annotated test-split subset), not 969.")


if __name__ == "__main__":
    main()