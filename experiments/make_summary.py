#!/usr/bin/env python3
"""make_summary.py -- build results/SUMMARY_tables.md from the JSON result files only
(no number is typed by hand). Usage: python experiments/make_summary.py"""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
SETS = ("888", "938")
BB = (("mplug", "mPLUG(-Owl2)"), ("blip", "BLIP-2"), ("git", "GIT"))


def J(*p):
    f = os.path.join(R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def t3(x):
    return " / ".join(f"{v:.1f}" for v in x)


def p(x):
    return f"{x:.3g}"


out = []
w = out.append

# ---------------------------------------------------------------- Table 5 check
w("## 1. Table 5 calibration (`check_table5.json`)\n")
w("R@10 / R@5 / R@1. 888 = Table 5 set (calibration target, must match); 938 = same logic + Assault.\n")
w("| Backbone | Set | VEZA | E-VEZA | Table 5 VEZA | Table 5 E-VEZA | frames kept % | E-VEZA videos w/o caption |")
w("|---|---|---|---|---|---|---|---|")
for s in SETS:
    c = J(s, "check_table5.json")
    if not c:
        continue
    for k, name in BB:
        r = c[k]
        w(f"| {name} | {s} (N={c['n_videos']}) | {t3(r['got']['veza'])} | {t3(r['got']['eveza'])} | "
          f"{t3(r['expected']['veza'])} | {t3(r['expected']['eveza'])} | {r['frames_ratio_pct']} | {r['eveza_videos_without_caption']} |")
w("")

# ---------------------------------------------------------------- timing
t = J("timing.json")
if t:
    w("## 2. Timing (`timing.json`)\n")
    w("| Quantity | Value |\n|---|---|")
    w(f"| Machine | {t['machine_note']} |")
    w(f"| MetaCLIP model | {t['clip']['model']} ({t['clip']['precision']}) |")
    w(f"| MetaCLIP image encoder, batch 1 (incl. preprocessing) | {1000*t['clip']['s_per_frame_batch1']:.2f} ms/frame |")
    w(f"| MetaCLIP image encoder, batch 32 (incl. preprocessing) | {1000*t['clip']['s_per_frame_batch32']:.2f} ms/frame |")
    w(f"| MetaCLIP peak GPU memory (torch allocated) | {t['clip']['peak_gpu_MiB']:.1f} MiB |")
    d = t["decode_s_per_frame"]
    w(f"| Seek + decode one 3 s-grid frame (OpenCV) | mean {1000*d['mean']:.2f} ms, median {1000*d['median']:.2f} ms (n={d['n']}) |")
    g = t["gate_s_per_video"]
    w(f"| Gate per video (load npz + B=200 histogram + median/MAD) | mean {g['mean']:.3f} s, median {g['median']:.3f} s (n={g['n']}) |")
    w(f"| Break-even recogniser latency L* = c_g/((1-r)N) - d, N=37.7, r=0.284 | {t['break_even_L_ms']} ms/frame |")
    w("")

# ---------------------------------------------------------------- bins
w("## 3. B sweep end to end, mPLUG captions (`bins_endtoend.json`)\n")
for s in SETS:
    b = J(s, "bins_endtoend.json")
    if not b:
        continue
    w(f"**{s} videos (N={b['n_videos']})** — VEZA (all frames): R@10/5/1 {t3(b['veza_full']['rak'])}, "
      f"{b['veza_full']['captions_per_video']} captions/video. Recomputed B=200 windows identical to the published ones for "
      f"{b.get('B200_windows_identical_to_published', 'n/a')}/{b['n_videos']} videos.\n")
    w("| B | R@10 / R@5 / R@1 | captions/video | % of VEZA frames | windows/video | no-caption videos | R@10 95% Wilson | McNemar vs B200 R@1 (B200-only / this-only / p) | R@5 | R@10 |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for B in (50, 100, 200, 500):
        r = b.get(f"B{B}")
        if not r:
            continue
        m = [r[f"mcnemar_R{K}_vs_B200"] for K in (1, 5, 10)]
        mm = [f"{x['B200_only']}/{x['this_only']}/{p(x['p'])}" for x in m]
        w(f"| {B} | {t3(r['rak'])} | {r['captions_per_video']} | {r['frames_pct_of_veza']} | {r['windows_per_video']} | "
          f"{r['no_caption_videos']} | {r['R10_wilson'][0]}–{r['R10_wilson'][1]} | {mm[0]} | {mm[1]} | {mm[2]} |")
    w("")

# ---------------------------------------------------------------- blip controls
w("## 4. Backbone controls (`blip_controls.json`)\n")
for s in SETS:
    b = J(s, "blip_controls.json")
    if not b:
        continue
    w(f"**{s} videos (N={b['n_videos']})**\n")
    w("| | " + " | ".join(n for _, n in BB) + " |")
    w("|---|" + "---|" * len(BB))
    row = lambda lab, f: w(f"| {lab} | " + " | ".join(f(b[k]) for k, _ in BB) + " |")
    row("VEZA R@10/5/1", lambda r: t3(r["veza"]))
    row("E-VEZA R@10/5/1", lambda r: t3(r["eveza"]))
    for K in (1, 5, 10):
        row(f"McNemar R@{K} (VEZA-only / E-VEZA-only / p)",
            lambda r, K=K: f"{r[f'mcnemar_R{K}']['veza_only']} / {r[f'mcnemar_R{K}']['eveza_only']} / {p(r[f'mcnemar_R{K}']['p'])}")
    row("captions/video VEZA → E-VEZA", lambda r: f"{r['captions_per_video']['veza']:.1f} → {r['captions_per_video']['eveza']:.1f}")
    row("median tokens VEZA / E-VEZA", lambda r: f"{r['tokens_veza_median']:.0f} / {r['tokens_eveza_median']:.0f}")
    row("% docs truncated at 512 VEZA / E-VEZA", lambda r: f"{r['pct_truncated_veza']} / {r['pct_truncated_eveza']}")
    row("sampling interval median (s); videos ≤1.5 s / ≥2.5 s",
        lambda r: f"{r['sampling_interval_s']['median']}; {r['sampling_interval_s']['videos_at_<=1.5s']} / {r['sampling_interval_s']['videos_at_>=2.5s']}")
    row("random length-matched, 10 seeds, mean (R@10/5/1)", lambda r: t3(r["random_matched"]["mean"]))
    row("  sd", lambda r: t3(r["random_matched"]["sd"]))
    row("  min / max R@1", lambda r: f"{r['random_matched']['min'][2]:.1f} / {r['random_matched']['max'][2]:.1f}")
    row("no-truncation (chunked) VEZA", lambda r: t3(r["chunked_veza"]))
    row("no-truncation (chunked) E-VEZA", lambda r: t3(r["chunked_eveza"]))
    row("chunked McNemar R@1 (VEZA-only / E-VEZA-only / p)",
        lambda r: f"{r['chunked_mcnemar_R1']['veza_only']} / {r['chunked_mcnemar_R1']['eveza_only']} / {p(r['chunked_mcnemar_R1']['p'])}")
    w("")

# ---------------------------------------------------------------- clip gate
w("## 5. Event gate + MetaCLIP (`clip_gate.json`)\n")
for s in SETS:
    c = J(s, "clip_gate.json")
    if not c:
        continue
    w(f"**{s} videos (N={c['n_videos']})**, model {c['model']}, 13 anomaly classes, empty gate = miss\n")
    w("| Aggregation | Arm | Top-1 / 3 / 5 | frames/video | % of all frames | videos with no gated frame | McNemar Top-1 vs all (all-only / gated-only / p) | McNemar Top-5 | Top-1 Wilson | Top-5 Wilson |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for agg in ("vote", "mean"):
        for arm in ("all", "gated_B200", "gated_B500"):
            r = c.get(f"{agg}/{arm}")
            if not r:
                continue
            if arm == "all":
                w(f"| {agg} | {arm} | {t3(r['top1_3_5'])} | {r['frames_per_video']} | 100 | – | – | – | – | – |")
            else:
                m1, m5 = r["mcnemar_top1_vs_all"], r["mcnemar_top5_vs_all"]
                w(f"| {agg} | {arm} | {t3(r['top1_3_5'])} | {r['frames_per_video']} | {r['frames_pct_of_all']} | "
                  f"{r['videos_with_no_gated_frame']} | {m1['all_only']}/{m1['gated_only']}/{p(m1['p'])} | "
                  f"{m5['all_only']}/{m5['gated_only']}/{p(m5['p'])} | {r['top1_wilson'][0]}–{r['top1_wilson'][1]} | "
                  f"{r['top5_wilson'][0]}–{r['top5_wilson'][1]} |")
    w("")
    if s == "938":
        w("Prompts (per class, mean of 5 normalised text embeddings):\n")
        for k, v in c["prompts"].items():
            w(f"- {k}: " + "; ".join(v))
        w("")

# ---------------------------------------------------------------- threshold ablation
w("## 6. Threshold ablation, paper Table 12 (`ablation.json`)\n")
w("mPLUG captions, slack rule, B=200, windows recomputed from the event-stream histograms; same CONFIG as the check. "
  "**edges rel** = `variant_rak.py` convention (bin edges from the first event, reproduces rak_final.txt); "
  "**edges abs** = `np.histogram` absolute edges as in `run_temporal_selection.py` (MAD/abs = the published windows). "
  "McNemar vs MAD within the same convention (MAD-only / this-only / p).\n")
for s in SETS:
    a = J(s, "ablation.json")
    if not a:
        continue
    hs = a["histogram_source"]; v = a["cache_verification"]
    w(f"**{s} videos (N={a['n_videos']})**: VEZA {t3(a['veza_full']['rak'])}; published windows {t3(a['published_windows']['rak'])}. "
      f"Histograms: {hs['variant_counts_cache']} from `variant_counts_cache.npz`, {hs['loaded_from_npz']} loaded from npz; "
      f"cache re-verified against full stream reloads {v['counts_identical']}/{v['n']}. "
      f"abs-edge MAD windows identical to published: {a['abs_MAD_windows_identical_to_published']}/{a['n_videos']}.\n")
    w("| Edges | Threshold | R@10 / R@5 / R@1 | captions/video | % of VEZA | no-caption videos | windows/video | McNemar R@1 vs MAD | McNemar R@5 vs MAD | McNemar R@10 vs MAD |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    names = {"MAD": "median + 2·MAD", "Mean": "mean + 1σ", "Fixed": "80th percentile", "Median": "median"}
    for conv in ("rel", "abs"):
        for var in ("MAD", "Mean", "Fixed", "Median"):
            r = a[f"edges_{conv}"][var]
            m = [r[f"mcnemar_R{K}_vs_MAD"] for K in (1, 5, 10)]
            mm = [f"{x['MAD_only']}/{x['this_only']}/{p(x['p'])}" for x in m]
            w(f"| {conv} | {names[var]} | {t3(r['rak'])} | {r['captions_per_video']} | {r['frames_pct_of_veza']} | "
              f"{r['no_caption_videos']} | {r['windows_per_video']} | {mm[0]} | {mm[1]} | {mm[2]} |")
    if "rak_final_reproduction" in a:
        rr = a["rak_final_reproduction"]
        w(f"\nReproduction of `rak_final.txt` (slack, edges rel): **{'all four variants match' if rr['all_match'] else 'MISMATCH'}** "
          "(R@10/5/1, no-caption videos and captions/video).\n")
        w("| Threshold | got | rak_final.txt | no-caption got / rak_final | captions/video got / rak_final |")
        w("|---|---|---|---|---|")
        for var in ("MAD", "Mean", "Fixed", "Median"):
            c = rr[var]
            w(f"| {names[var]} | {t3(c['got'])} | {t3(c['rak_final'])} | {c['no_caption_got']} / {c['no_caption_rak_final']} | "
              f"{c['cpv_got']} / {c['cpv_rak_final']} |")
    w("")

open(os.path.join(R, "SUMMARY_tables.md"), "w").write("\n".join(out) + "\n")
print("[SAVED] results/SUMMARY_tables.md")
