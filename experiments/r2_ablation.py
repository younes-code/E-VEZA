#!/usr/bin/env python3
"""
r2_ablation.py -- threshold ablation (paper Table 12), mPLUG captions, slack rule, same CONFIG as `r2_text.py check`.

Thresholds over B=200 equal bins of event timestamps (copied from scripts_rev/variant_rak.py `thresholds`):
  MAD    median + 2*MAD        Mean   mean + 1*std (numpy, ddof=0)
  Fixed  80th percentile       Median median only
Active bins: strict '>' ; adjacent active bins merged (variant_rak `windows_from`).

Two bin-edge conventions (identical counts, different time origin):
  rel  edges = linspace(0, T, B+1), T = t.max()-t.min()   -> variant_rak.py (rak_final.txt). PRIMARY.
  abs  edges = t.min() + linspace(0, T, B+1)              -> np.histogram edges, as in scripts/run_temporal_selection.py
       (MAD/abs must reproduce the published windows exactly).

Histograms: variant_counts_cache.npz (written by variant_rak.py from the streams: np.histogram(t/1e6, bins=200)) for the
videos it holds; every other video (Assault) is loaded from its npz. t.min() is read from the start of t.npy (t is sorted)
for every video. --verify N reloads N random cached videos in full and checks counts, T and t.min.

  python experiments/r2_ablation.py --eval-list eval888.txt --out results/888
"""
import argparse, os, random, sys, time, zipfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2_common as C

B = 200
VARIANTS = ("MAD", "Mean", "Fixed", "Median")
RAK_FINAL_888 = {"MAD": (88.7, 66.0, 27.3), "Mean": (87.3, 63.2, 26.9),
                 "Fixed": (89.2, 66.2, 26.7), "Median": (89.4, 66.0, 27.5)}
RAK_FINAL_888_NOCAP = {"MAD": 1, "Mean": 17, "Fixed": 2, "Median": 0}
RAK_FINAL_888_CPV = {"MAD": 10.7, "Mean": 6.2, "Fixed": 10.7, "Median": 22.5}


def thresholds(c):
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med)))
    return {"MAD": med + 2.0 * mad, "Mean": float(c.mean() + c.std()),
            "Fixed": float(np.percentile(c, 80)), "Median": med}


def windows_from(c, thr, edges):
    active = np.flatnonzero(c > thr)
    if active.size == 0:
        return []
    out, start, prev = [], active[0], active[0]
    for b in active[1:]:
        if b == prev + 1:
            prev = b
            continue
        out.append((float(edges[start]), float(edges[prev + 1])))
        start = prev = b
    out.append((float(edges[start]), float(edges[prev + 1])))
    return out


def t_first(npz_path):
    """First element of t.npy (sorted => t.min()) without decompressing the whole array."""
    with zipfile.ZipFile(npz_path) as z, z.open("t.npy") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        return float(np.frombuffer(f.read(dtype.itemsize), dtype=dtype)[0])


def full_hist(npz_path):
    t = np.load(npz_path)["t"] / 1e6
    c, _ = np.histogram(t, bins=B)
    return c.astype(float), float(t.max() - t.min()), float(t.min())


def main(a):
    vids = sorted(l.strip() for l in open(a.eval_list) if l.strip())
    npz = C.find_npz(a.npz_root)
    cache = {}
    if os.path.exists(a.cache):  # optional: histograms cached by evaluation/variant_rak.py
        d = np.load(a.cache, allow_pickle=True)
        cache = dict(zip(d["names"].tolist(), zip(d["counts"].astype(float), d["T"].astype(float))))
    hist, src, t0 = {}, {}, time.time()
    for i, v in enumerate(vids):
        if v in cache:
            c, T = cache[v]
            hist[v] = (c, float(T), t_first(npz[v]) / 1e6)
            src[v] = "cache"
        else:
            hist[v] = full_hist(npz[v])
            src[v] = "stream"
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(vids)} histograms ({time.time()-t0:.0f}s)", flush=True)
    n_stream = sum(s == "stream" for s in src.values())
    print(f"[INFO] {len(vids)} videos: {len(vids)-n_stream} histograms from variant_counts_cache.npz, {n_stream} loaded from streams")

    # verification of cached histograms against full stream loads
    ver = {"n": 0, "counts_identical": 0, "max_abs_dT": 0.0, "max_abs_dtmin": 0.0, "videos": []}
    cached = [v for v in vids if src[v] == "cache"]
    random.seed(0)
    for v in random.sample(cached, min(a.verify, len(cached))):
        c, T, tmin = full_hist(npz[v])
        ver["n"] += 1
        ver["counts_identical"] += int(np.array_equal(c, hist[v][0]))
        ver["max_abs_dT"] = max(ver["max_abs_dT"], abs(T - hist[v][1]))
        ver["max_abs_dtmin"] = max(ver["max_abs_dtmin"], abs(tmin - hist[v][2]))
        ver["videos"].append(v)
    print(f"[VERIFY] cached histograms identical to a full stream reload: {ver['counts_identical']}/{ver['n']} "
          f"(max |dT| {ver['max_abs_dT']:.3g} s, max |d tmin| {ver['max_abs_dtmin']:.3g} s)")

    caps = C.load_captions(os.path.join(a.captions_dir, "mplug_captions.txt"))
    pub = C.load_windows(a.windows)
    ts = C.TextScorer(device=a.device)

    def score(wins):
        docs, empty_flags, n = [], [], []
        for v in vids:
            sel = caps[v] if wins is None else C.select(caps[v], wins[v])
            docs.append(ts.doc(sel)); empty_flags.append(len(sel) == 0); n.append(len(sel))
        r = ts.true_rank(ts.embed_docs(docs), vids)
        if C.EMPTY_DOC_IS_MISS:
            r = np.where(np.array([dd == "" for dd in docs]), 99, r)
        return r, int(sum(empty_flags)), float(np.mean(n))

    out = {"n_videos": len(vids), "eval_list": a.eval_list, "B": B, "backbone": "mplug", "rule": "slack, delta=1 s",
           "config": {"TEXT_MODEL": C.TEXT_MODEL, "MAX_SEQ_LEN": C.MAX_SEQ_LEN, "STRIP_EOS": C.STRIP_EOS,
                      "EMPTY_DOC_IS_MISS": C.EMPTY_DOC_IS_MISS, "DESCRIPTOR_JOIN": C.DESCRIPTOR_JOIN},
           "histogram_source": {"variant_counts_cache": len(vids) - n_stream, "loaded_from_npz": n_stream},
           "cache_verification": ver}
    rv, _, nv = score(None)
    out["veza_full"] = {"rak": C.rak(rv), "captions_per_video": round(nv, 2)}
    rp, ep, npub = score(pub)
    out["published_windows"] = {"rak": C.rak(rp), "captions_per_video": round(npub, 2), "no_caption_videos": ep}
    print(f"VEZA {out['veza_full']['rak']}   published windows {out['published_windows']['rak']}")

    for conv in ("rel", "abs"):
        res, ranks = {}, {}
        for var in VARIANTS:
            wins = {}
            for v in vids:
                c, T, tmin = hist[v]
                edges = np.linspace(0.0, T, B + 1) + (tmin if conv == "abs" else 0.0)
                wins[v] = windows_from(c, thresholds(c)[var], edges)
            if conv == "abs" and var == "MAD":
                same = sum(len(wins[v]) == len(pub.get(v, [])) and
                           np.allclose(np.array(wins[v]).reshape(-1, 2), np.array(pub.get(v, [])).reshape(-1, 2))
                           for v in vids)
                out["abs_MAD_windows_identical_to_published"] = int(same)
                print(f"[SANITY] abs-edge MAD windows identical to published for {same}/{len(vids)} videos")
            r, empty, cpv = score(wins)
            ranks[var] = r
            res[var] = {"rak": C.rak(r), "captions_per_video": round(cpv, 2),
                        "frames_pct_of_veza": round(100 * cpv / nv, 1), "no_caption_videos": empty,
                        "windows_per_video": round(float(np.mean([len(wins[v]) for v in vids])), 2),
                        "R10_wilson": C.wilson(int(np.sum(r <= 10)), len(vids))}
        for var in VARIANTS:
            for K in (1, 5, 10):
                b, c_, p = C.mcnemar_exact(ranks["MAD"] <= K, ranks[var] <= K)
                res[var][f"mcnemar_R{K}_vs_MAD"] = {"MAD_only": b, "this_only": c_, "p": p}
        out[f"edges_{conv}"] = res
        print(f"\n== edges {conv}")
        for var in VARIANTS:
            r = res[var]
            line = (f"  {var:<7} R@10/5/1 {r['rak']}  captions/video {r['captions_per_video']:.2f}  no-caption {r['no_caption_videos']:3d}  "
                    f"McNemar vs MAD R@1 {r['mcnemar_R1_vs_MAD']['MAD_only']}/{r['mcnemar_R1_vs_MAD']['this_only']} p={r['mcnemar_R1_vs_MAD']['p']:.3g}  "
                    f"R@5 {r['mcnemar_R5_vs_MAD']['MAD_only']}/{r['mcnemar_R5_vs_MAD']['this_only']} p={r['mcnemar_R5_vs_MAD']['p']:.3g}")
            print(line)

    if len(vids) == 888:
        chk, ok = {}, True
        for var in VARIANTS:
            got = out["edges_rel"][var]
            diff = max(abs(x - y) for x, y in zip(got["rak"], RAK_FINAL_888[var]))
            good = (diff <= 0.05 and got["no_caption_videos"] == RAK_FINAL_888_NOCAP[var]
                    and round(got["captions_per_video"], 1) == RAK_FINAL_888_CPV[var])
            ok &= good
            chk[var] = {"got": got["rak"], "rak_final": RAK_FINAL_888[var], "no_caption_got": got["no_caption_videos"],
                        "no_caption_rak_final": RAK_FINAL_888_NOCAP[var], "cpv_got": round(got["captions_per_video"], 1),
                        "cpv_rak_final": RAK_FINAL_888_CPV[var], "match": bool(good)}
            print(f"{'OK ' if good else 'MISMATCH'} {var:<7} got {got['rak']} nocap {got['no_caption_videos']} cpv {got['captions_per_video']:.1f}"
                  f"   rak_final {RAK_FINAL_888[var]} nocap {RAK_FINAL_888_NOCAP[var]} cpv {RAK_FINAL_888_CPV[var]}")
        out["rak_final_reproduction"] = {"all_match": bool(ok), **chk}
        print("rak_final.txt REPRODUCED" if ok else "rak_final.txt NOT reproduced")
    C.save_json(out, os.path.join(a.out, "ablation.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-list", required=True)
    ap.add_argument("--captions-dir", default="outputs/captions/full")
    ap.add_argument("--windows", default="outputs/windows/global_active_windows.json")
    ap.add_argument("--npz-root", default="data/UCF-Crime-DVS")
    ap.add_argument("--cache", default="variant_counts_cache.npz")
    ap.add_argument("--verify", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
