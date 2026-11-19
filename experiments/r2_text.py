"""
r2_text.py  --  text-stage experiments for MTAP round 2 (CPU or GPU, no VLM needed).

  python r2_text.py check   # STEP 1: must reproduce Table 5 before anything else
  python r2_text.py bins    # R8-3: end-to-end R@K and frames/video for B in {50,100,200,500}
  python r2_text.py blip    # R8-4: McNemar, length-matched random control, no-truncation control,
                            #       truncation statistics, caption sampling interval

Common options:
  --captions-dir outputs/captions/full                    folder with mplug_/blip_/git_captions.txt (full, VEZA)
  --windows outputs/windows/global_active_windows.json   published E-VEZA windows (B=200)
  --npz-root data/UCF-Crime-DVS            event streams (only needed by `bins`)
  --eval-list eval888.txt                  optional: one video name per line (the 888 of Table 5)
  --out results
"""
import argparse
import os
import time

import numpy as np

import r2_common as C

BACKBONES = {"mplug": "mplug_captions.txt", "blip": "blip_captions.txt", "git": "git_captions.txt"}


def eval_set(args, caps_by_bb, windows):
    if args.eval_list and os.path.exists(args.eval_list):
        vids = [l.strip() for l in open(args.eval_list) if l.strip()]
        print(f"[INFO] eval list from {args.eval_list}: {len(vids)} videos")
        return sorted(vids)
    common = set(windows)
    for caps in caps_by_bb.values():
        common &= set(caps)
    common = {v for v in common if C.video_class(v) in C.ANOMALY_CLASSES}
    print(f"[INFO] eval set = videos with windows and captions for all backbones: {len(common)}")
    if len(common) != 888:
        print("[WARN] not 888. Dump the exact list from variant_rak.py into eval888.txt and pass --eval-list.")
    return sorted(common)


def ranks_for(ts, caps, vids, windows=None, chunked=False):
    docs, empty = [], 0
    for v in vids:
        sel = caps[v] if windows is None else C.select(caps[v], windows.get(v, []))
        empty += (len(sel) == 0)
        docs.append(ts.doc(sel))
    ranks = ts.true_rank(ts.embed_docs(docs, chunked=chunked), vids)
    if C.EMPTY_DOC_IS_MISS:
        ranks = np.where(np.array([d == "" for d in docs]), 99, ranks)
    return ranks, docs, empty


def load_all(args):
    caps = {bb: C.load_captions(os.path.join(args.captions_dir, f)) for bb, f in BACKBONES.items()}
    wins = C.load_windows(args.windows)
    vids = eval_set(args, caps, wins)
    return caps, wins, vids


# ----------------------------------------------------------------------------------
def cmd_check(args):
    caps, wins, vids = load_all(args)
    ts = C.TextScorer(device=args.device)
    ok, report = True, {}
    for bb in BACKBONES:
        rv, _, _ = ranks_for(ts, caps[bb], vids)
        re_, _, empty = ranks_for(ts, caps[bb], vids, wins)
        got = {"veza": C.rak(rv), "eveza": C.rak(re_)}
        exp = C.EXPECTED_TABLE5[bb]
        for arm in ("veza", "eveza"):
            diff = max(abs(a - b) for a, b in zip(got[arm], exp[arm]))
            flag = "OK " if diff <= 0.15 else "MISMATCH"
            ok &= diff <= 0.15
            print(f"{flag} {bb:5s} {arm:5s} got {got[arm]}  expected {exp[arm]}")
        n_v = sum(len(caps[bb][v]) for v in vids)
        n_e = sum(len(C.select(caps[bb][v], wins.get(v, []))) for v in vids)
        report[bb] = {"got": got, "expected": exp, "frames_ratio_pct": round(100 * n_e / n_v, 1),
                      "eveza_videos_without_caption": empty}
        print(f"     {bb:5s} frames kept {report[bb]['frames_ratio_pct']} %   no-caption videos {empty}")
    report["n_videos"] = len(vids)
    C.save_json(report, os.path.join(args.out, "check_table5.json"))
    with open(os.path.join(args.out, "eval_videos.txt"), "w") as f:
        f.write("\n".join(vids))
    print("\nALL OK, go on." if ok else "\nSTOP: adjust CONFIG in r2_common.py (or --eval-list) until Table 5 reproduces.")


# ----------------------------------------------------------------------------------
def cmd_bins(args):
    caps, wins_pub, vids = load_all(args)
    npz = C.find_npz(args.npz_root)
    ts = C.TextScorer(device=args.device)
    bb = "mplug"
    rv, _, _ = ranks_for(ts, caps[bb], vids)
    n_veza = np.mean([len(caps[bb][v]) for v in vids])
    rows = {"veza_full": {"rak": C.rak(rv), "captions_per_video": round(float(n_veza), 2)}}
    missing = [v for v in vids if v not in npz]
    print(f"[INFO] {len(missing)} eval videos have no readable npz (they keep the published windows)")
    wcache = C.windows_cache(npz, vids, args.bins, args.windows_cache)
    for B in args.bins:
        t0 = time.time()
        wins = {}
        for v in vids:
            got = wcache.get(v, {}).get(str(B))
            if got is None:   # no / unreadable stream -> published windows (as before)
                print(f"[SKIP] {v}: no windows for B={B}, using published"); wins[v] = wins_pub.get(v, [])
            else:
                wins[v] = [tuple(w) for w in got]
        if B == 200:
            same = sum(np.allclose(np.array(wins[v]).reshape(-1, 2), np.array(wins_pub.get(v, [])).reshape(-1, 2))
                       if len(wins[v]) == len(wins_pub.get(v, [])) else False for v in vids)
            print(f"[SANITY] recomputed B=200 windows identical to published for {same}/{len(vids)} videos")
            n_same_b200 = int(same)
        r, _, empty = ranks_for(ts, caps[bb], vids, wins)
        n = np.mean([len(C.select(caps[bb][v], wins[v])) for v in vids])
        k10 = int(np.sum(r <= 10))
        rows[f"B{B}"] = {"rak": C.rak(r), "captions_per_video": round(float(n), 2),
                         "frames_pct_of_veza": round(100 * n / n_veza, 1),
                         "windows_per_video": round(float(np.mean([len(wins[v]) for v in vids])), 2),
                         "no_caption_videos": empty, "R10_wilson": C.wilson(k10, len(vids)),
                         "mcnemar_R1_vs_B200": None}
        rows[f"B{B}"]["_ranks"] = r.tolist()
        print(f"B={B:4d}  R@10/5/1 {rows[f'B{B}']['rak']}  captions/video {n:.2f} "
              f"({rows[f'B{B}']['frames_pct_of_veza']} % of VEZA)  {time.time()-t0:.0f}s")
    if "B200" in rows:
        for B in args.bins:
            for K in (1, 5, 10):
                b, c, p = C.mcnemar_exact(np.array(rows["B200"]["_ranks"]) <= K, np.array(rows[f"B{B}"]["_ranks"]) <= K)
                rows[f"B{B}"][f"mcnemar_R{K}_vs_B200"] = {"B200_only": b, "this_only": c, "p": p}
    for k in list(rows):
        rows[k].pop("_ranks", None)
    rows["n_videos"] = len(vids)
    if 200 in args.bins:
        rows["B200_windows_identical_to_published"] = n_same_b200
    C.save_json(rows, os.path.join(args.out, "bins_endtoend.json"))


# ----------------------------------------------------------------------------------
def cmd_blip(args):
    caps, wins, vids = load_all(args)
    ts = C.TextScorer(device=args.device)
    rng = np.random.default_rng(0)
    out = {"n_videos": len(vids)}
    for bb in BACKBONES:
        t0 = time.time()
        res = {}
        rv, dv, _ = ranks_for(ts, caps[bb], vids)
        re_, de, _ = ranks_for(ts, caps[bb], vids, wins)
        res["veza"], res["eveza"] = C.rak(rv), C.rak(re_)
        # 1. paired McNemar at every depth
        for K in (1, 5, 10):
            b, c, p = C.mcnemar_exact(rv <= K, re_ <= K)
            res[f"mcnemar_R{K}"] = {"veza_only": b, "eveza_only": c, "p": p}
        # 2. truncation statistics
        tv = np.array([ts.n_tokens(d) for d in dv]); te = np.array([ts.n_tokens(d) for d in de])
        res["tokens_veza_median"] = float(np.median(tv)); res["tokens_eveza_median"] = float(np.median(te))
        res["pct_truncated_veza"] = round(100 * float(np.mean(tv > C.MAX_SEQ_LEN)), 1)
        res["pct_truncated_eveza"] = round(100 * float(np.mean(te > C.MAX_SEQ_LEN)), 1)
        # 3. sampling interval of the caption file
        gaps = np.array([C.sampling_interval(caps[bb][v]) for v in vids])
        res["sampling_interval_s"] = {"median": float(np.nanmedian(gaps)),
                                      "videos_at_<=1.5s": int(np.sum(gaps <= 1.5)),
                                      "videos_at_>=2.5s": int(np.sum(gaps >= 2.5))}
        res["captions_per_video"] = {"veza": float(np.mean([len(caps[bb][v]) for v in vids])),
                                     "eveza": float(np.mean([len(C.select(caps[bb][v], wins.get(v, []))) for v in vids]))}
        # 4. length-matched random control: same number of captions as E-VEZA, drawn at random
        k_sel = {v: len(C.select(caps[bb][v], wins.get(v, []))) for v in vids}
        seeds = []
        for s in range(args.seeds):
            docs = []
            for v in vids:
                pool = caps[bb][v]
                idx = np.sort(rng.choice(len(pool), size=min(k_sel[v], len(pool)), replace=False)) if pool else []
                docs.append(ts.doc([pool[i] for i in idx]))
            r = ts.true_rank(ts.embed_docs(docs), vids)
            seeds.append(C.rak(r))
        seeds = np.array(seeds)
        res["random_matched"] = {"mean": seeds.mean(0).round(1).tolist(), "sd": seeds.std(0).round(1).tolist(),
                                 "min": seeds.min(0).tolist(), "max": seeds.max(0).tolist()}
        # 5. no-truncation control: chunked mean-pooled encoding of the same documents
        rvc, _, _ = ranks_for(ts, caps[bb], vids, chunked=True)
        rec, _, _ = ranks_for(ts, caps[bb], vids, wins, chunked=True)
        res["chunked_veza"], res["chunked_eveza"] = C.rak(rvc), C.rak(rec)
        for K in (1,):
            b, c, p = C.mcnemar_exact(rvc <= K, rec <= K)
            res[f"chunked_mcnemar_R{K}"] = {"veza_only": b, "eveza_only": c, "p": p}
        out[bb] = res
        print(f"\n== {bb} ({time.time()-t0:.0f}s)\n"
              f" VEZA {res['veza']}  E-VEZA {res['eveza']}   McNemar R@1 p={res['mcnemar_R1']['p']:.3g}\n"
              f" truncated: VEZA {res['pct_truncated_veza']} %  E-VEZA {res['pct_truncated_eveza']} %"
              f"   median tokens {res['tokens_veza_median']:.0f} / {res['tokens_eveza_median']:.0f}\n"
              f" random length-matched ({args.seeds} seeds) {res['random_matched']['mean']} +- {res['random_matched']['sd']}\n"
              f" no truncation: VEZA {res['chunked_veza']}  E-VEZA {res['chunked_eveza']}\n"
              f" sampling interval {res['sampling_interval_s']}")
    C.save_json(out, os.path.join(args.out, "blip_controls.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "bins", "blip"])
    ap.add_argument("--captions-dir", default="outputs/captions/full")
    ap.add_argument("--windows", default="outputs/windows/global_active_windows.json")
    ap.add_argument("--npz-root", default="data/UCF-Crime-DVS")
    ap.add_argument("--eval-list", default="")
    ap.add_argument("--bins", type=int, nargs="+", default=[50, 100, 200, 500])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--windows-cache", default="results/windows_by_B.json")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"check": cmd_check, "bins": cmd_bins, "blip": cmd_blip}[a.cmd](a)
