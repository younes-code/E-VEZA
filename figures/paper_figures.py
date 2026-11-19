"""
paper_figures.py
================
Generate all paper-quality figures for E-VEZA from:
  - global_active_windows.json   (output of script 1 / process_all)

Usage
-----
    python paper_figures.py \
        --json  outputs/windows/global_active_windows.json \
        --out   paper_figures

Produces (in --out folder):
  fig1_coverage_barplot.pdf/.png   — per-class avg coverage with std error bars
  fig2_sensitivity.pdf/.png        — coverage vs MAD scale factor (threshold sensitivity)
  fig3_duration_vs_coverage.pdf/.png — avg video duration vs avg coverage scatter
  fig4_coverage_distribution.pdf/.png — violin/box of coverage % across all 969 videos
  fig5_histogram_sample.pdf/.png   — single-video event histogram (best sample)

Extra modes
-----------
  --only fig1|fig2|fig3|fig4|fig5   regenerate a single figure (fast)
  --only gallery                    render one histogram per readable video
                                    into <out>/gallery/ and print a table
                                    ranked by contrast, so a good sample is
                                    easy to pick. Then render the winner:
                                    --only fig5 --hist-class C --hist-video V

All figures use a consistent style suitable for IEEE two-column format.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from glob import glob
import zipfile

# ─── Shared style ─────────────────────────────────────────────────────────────

PALETTE = {
    "primary":   "#4361ee",
    "accent":    "#f72585",
    "highlight": "#f59e0b",
    "neutral":   "#6b7280",
    "light":     "#e5e7eb",
    "green":     "#10b981",
    "bg":        "white",
}

CLASS_ORDER = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "RoadAccidents", "Robbery",
    "Shooting", "Shoplifting", "Stealing", "Vandalism",
]

# Display labels (shorter, for axis ticks)
CLASS_LABELS = {
    "Abuse": "Abuse",
    "Arrest": "Arrest",
    "Arson": "Arson",
    "Assault": "Assault",
    "Burglary": "Burglary",
    "Explosion": "Explosion",
    "Fighting": "Fighting",
    "RoadAccidents": "Road Acc.",
    "Robbery": "Robbery",
    "Shooting": "Shooting",
    "Shoplifting": "Shoplifting",
    "Stealing": "Stealing",
    "Vandalism": "Vandalism",
}

def set_ieee_style():
    """Apply a clean, IEEE-compatible matplotlib style."""
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#e5e7eb",
        "grid.linewidth":    0.6,
        "axes.axisbelow":    True,
    })

# ─── Data loading ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def group_by_class(data):
    groups = defaultdict(list)
    for entry in data:
        cls = entry.get("class_name") or _class_from_path(entry.get("npz_path", ""))

        if cls:
            # 🔥 STRONG normalization
            cls = cls.strip().lower()

            # map to canonical names
            mapping = {
                "assault": "Assault",
                "abuse": "Abuse",
                "arson": "Arson",
                "arrest": "Arrest",
                "burglary": "Burglary",
                "explosion": "Explosion",
                "fighting": "Fighting",
                "roadaccidents": "RoadAccidents",
                "robbery": "Robbery",
                "shooting": "Shooting",
                "shoplifting": "Shoplifting",
                "stealing": "Stealing",
                "vandalism": "Vandalism",
            }

            cls = mapping.get(cls, cls)  # fallback if unknown
            groups[cls].append(entry)

    return groups

def _class_from_path(path):
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        return parts[2]
    return None

def per_class_stats(groups):
    """Return dict: class → {mean_cov, std_cov, mean_dur, n_videos, mean_windows}"""
    stats = {}
    for cls, records in groups.items():
        covs = [r["coverage_percent"] for r in records]
        durs = [r.get("total_duration_sec", 0) for r in records]
        n_wins = [len(r.get("active_windows", [])) for r in records]
        stats[cls] = {
            "mean_cov":     np.mean(covs),
            "std_cov":      np.std(covs),
            "mean_dur":     np.mean(durs),
            "n_videos":     len(records),
            "mean_windows": np.mean(n_wins),
        }
    return stats

# ─── Figure 1: Per-class coverage bar chart with error bars ───────────────────

def fig_coverage_barplot(stats, out_dir):
    """
    Horizontal bar chart: avg coverage (%) per class, error bars = std.
    Bars coloured by coverage magnitude (low→high = cool→warm).
    """
    set_ieee_style()

    # Use only classes present in data, respecting CLASS_ORDER
    classes = [c for c in CLASS_ORDER if c in stats]
    means   = [stats[c]["mean_cov"] for c in classes]
    stds    = [stats[c]["std_cov"]  for c in classes]
    labels  = [CLASS_LABELS.get(c, c) for c in classes]

    # Colour gradient: lowest coverage = blue, highest = orange
    norm_vals = np.array(means)
    norm_vals = (norm_vals - norm_vals.min()) / (norm_vals.max() - norm_vals.min() + 1e-9)
    cmap = plt.get_cmap("RdYlBu_r")
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    y_pos = np.arange(len(classes))
    bars = ax.barh(y_pos, means, xerr=stds,
                   color=colors, height=0.65,
                   error_kw=dict(ecolor=PALETTE["neutral"], lw=1.2, capsize=3),
                   zorder=3)

    # Global mean line
    global_mean = np.mean(means)
    ax.axvline(global_mean, color=PALETTE["accent"], linewidth=1.4,
               linestyle="--", zorder=4, label=f"Global mean ({global_mean:.1f}%)")

    # Value labels at end of each bar
    for bar, m, s in zip(bars, means, stds):
        ax.text(m + s + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{m:.1f}%", va="center", ha="left", fontsize=7.5,
                color=PALETTE["neutral"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Average Coverage (%)")
    ax.set_xlim(0, max(means) + max(stds) + 12)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.set_title("Per-Class Event-Guided Coverage", pad=8)

    fig.tight_layout()
    _save(fig, out_dir, "fig1_coverage_barplot")
    print("[DONE] fig1_coverage_barplot")

# ─── Figure 2: Threshold sensitivity ──────────────────────────────────────────

def fig_threshold_sensitivity(json_path, out_dir, n_sample=200, npz_base_dir=None):
    """
    Re-computes coverage for scale factors k ∈ [0.5, 3.5] from the NPZ event
    streams when --npz is given. Without it, falls back to the parametric
    approximation (kept below for reference only — do not publish it).
    """
    set_ieee_style()

    data = load_json(json_path)

    NPZ_BASE_DIR = npz_base_dir   # e.g.  "data/UCF-Crime-DVS"

    k_values = np.linspace(0.5, 3.5, 25)

    if NPZ_BASE_DIR and os.path.isdir(NPZ_BASE_DIR):
        print("[INFO] Recomputing sensitivity from NPZ files…")
        mean_covs, std_covs = _sensitivity_from_npz(NPZ_BASE_DIR, k_values, n_sample)
    else:
        print("[INFO] Using parametric sensitivity approximation…")
        mean_covs, std_covs = _sensitivity_parametric(data, k_values)

    fig, ax = plt.subplots(figsize=(3.8, 3.2))

    ax.plot(k_values, mean_covs, color=PALETTE["primary"],
            linewidth=2, zorder=3, label="Mean coverage")
    ax.fill_between(k_values,
                    np.array(mean_covs) - np.array(std_covs),
                    np.array(mean_covs) + np.array(std_covs),
                    alpha=0.18, color=PALETTE["primary"], zorder=2)

    # Mark k=2 (chosen value)
    idx2 = np.argmin(np.abs(k_values - 2.0))
    ax.scatter([k_values[idx2]], [mean_covs[idx2]],
               color=PALETTE["accent"], zorder=5, s=50,
               label="k=2 (ours)")
    ax.axvline(2.0, color=PALETTE["accent"], linewidth=1.0,
               linestyle=":", zorder=4)

    ax.set_xlabel("MAD scale factor  k")
    ax.set_ylabel("Mean Coverage (%)")
    ax.set_title(r"Threshold Sensitivity  ($\theta_k$ = Median + k·MAD)", pad=8)
    ax.legend(framealpha=0.85)
    ax.set_xlim(k_values[0], k_values[-1])

    fig.tight_layout()
    _save(fig, out_dir, "fig2_sensitivity")
    print("[DONE] fig2_sensitivity")

def _sensitivity_parametric(data, k_values):
    """
    FALLBACK ONLY — NOT FOR PUBLICATION.

    Rescales the single measured operating point (k=2) by a hand-chosen power
    law. It contains no information about k beyond what is already in the k=2
    number. Kept so the script still runs without --npz; always pass --npz for
    anything that goes in the paper.
    """
    coverages_at_k2 = np.array([r["coverage_percent"] for r in data])
    thresholds      = np.array([r.get("threshold", np.nan) for r in data])
    valid = ~np.isnan(thresholds) & (thresholds > 0)

    mean_covs, std_covs = [], []
    for k in k_values:
        scale = (k / 2.0) ** 0.55   # empirically calibrated exponent
        adjusted = coverages_at_k2 / scale
        adjusted = np.clip(adjusted, 0, 100)
        mean_covs.append(float(np.mean(adjusted)))
        std_covs.append(float(np.std(adjusted)))

    return mean_covs, std_covs

def _sensitivity_from_npz(base_dir, k_values, n_sample):
    """Exact recomputation by loading NPZ files."""
    npz_files = glob(os.path.join(base_dir, "*", "*.npz"))
    rng = np.random.default_rng(42)
    if len(npz_files) > n_sample:
        npz_files = rng.choice(npz_files, n_sample, replace=False).tolist()

    results = defaultdict(list)
    for path in npz_files:
        if not zipfile.is_zipfile(path):
            continue
        try:
            data_npz = np.load(path)
            t = data_npz["t"] / 1e6
            counts, bin_edges = np.histogram(t, bins=200)
            median_val = np.median(counts)
            mad = np.median(np.abs(counts - median_val))
            total = t.max() - t.min()
            if total <= 0:
                continue

            for k in k_values:
                thresh = median_val + k * mad
                active = counts > thresh
                duration = 0.0
                in_w = False
                for i, a in enumerate(active):
                    if a and not in_w:
                        w_start = bin_edges[i]; in_w = True
                    elif not a and in_w:
                        duration += bin_edges[i] - w_start; in_w = False
                if in_w:
                    duration += bin_edges[-1] - w_start
                results[k].append(duration / total * 100)
        except Exception:
            continue

    mean_covs = [np.mean(results[k]) if results[k] else np.nan for k in k_values]
    std_covs  = [np.std(results[k])  if results[k] else 0.0   for k in k_values]
    return mean_covs, std_covs

# ─── Figure 3: Avg video duration vs avg coverage scatter ─────────────────────

def fig_duration_vs_coverage(stats, out_dir):
    """
    Scatter plot: avg video duration (s) on x-axis (mean of
    total_duration_sec), avg coverage (%) on y-axis. Each dot = one class.
    """
    set_ieee_style()

    classes = [c for c in CLASS_ORDER if c in stats]
    x = [stats[c]["mean_dur"]  for c in classes]
    y = [stats[c]["mean_cov"]  for c in classes]
    n = [stats[c]["n_videos"]  for c in classes]
    labels = [CLASS_LABELS.get(c, c) for c in classes]

    # Dot size proportional to number of videos
    sizes = [max(30, v * 1.5) for v in n]

    # Colour by coverage (same gradient as fig1)
    norm_vals = (np.array(y) - min(y)) / (max(y) - min(y) + 1e-9)
    cmap = plt.get_cmap("RdYlBu_r")
    colors = [cmap(v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    sc = ax.scatter(x, y, s=sizes, c=colors, zorder=3, edgecolors="white",
                    linewidths=0.6, alpha=0.9)

    # Label each dot
    from matplotlib import patheffects
    for xi, yi, lbl in zip(x, y, labels):
        ax.annotate(lbl, (xi, yi),
                    textcoords="offset points", xytext=(4, 3),
                    fontsize=6.5, color="#374151",
                    path_effects=[patheffects.withStroke(linewidth=1.5,
                                                         foreground="white")])

    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    xs = np.linspace(min(x), max(x), 100)
    ax.plot(xs, p(xs), "--", color=PALETTE["neutral"],
            linewidth=1.0, alpha=0.6, label="Linear trend")

    ax.set_xlabel("Avg. Video Duration (s)")
    ax.set_ylabel("Avg. Coverage (%)")
    ax.set_title("Video Duration vs. Coverage per Class", pad=8)
    ax.set_xlim(min(x) - 8, max(x) * 1.15)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(y), vmax=max(y)))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Avg. Coverage (%)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    _save(fig, out_dir, "fig3_duration_vs_coverage")
    print("[DONE] fig3_duration_vs_coverage")

# ─── Figure 4: Coverage distribution violin + box ─────────────────────────────

def fig_coverage_distribution(groups, out_dir):
    """
    Violin + box plot showing the distribution of per-video coverage (%)
    for each class — reveals spread, not just the mean.
    """
    set_ieee_style()

    classes = [c for c in CLASS_ORDER if c in groups]
    data_per_class = [[r["coverage_percent"] for r in groups[c]] for c in classes]
    labels = [CLASS_LABELS.get(c, c) for c in classes]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))

    positions = np.arange(1, len(classes) + 1)

    # Violin
    vparts = ax.violinplot(data_per_class, positions=positions,
                           widths=0.7, showmedians=False,
                           showextrema=False)
    for pc in vparts["bodies"]:
        pc.set_facecolor(PALETTE["primary"])
        pc.set_alpha(0.35)
        pc.set_edgecolor(PALETTE["primary"])
        pc.set_linewidth(0.8)

    # Box (overlaid)
    bp = ax.boxplot(data_per_class, positions=positions,
                    widths=0.28, patch_artist=True,
                    medianprops=dict(color=PALETTE["accent"],
                                     linewidth=1.8),
                    boxprops=dict(facecolor="white",
                                  color=PALETTE["primary"],
                                  linewidth=1.0),
                    whiskerprops=dict(color=PALETTE["primary"],
                                      linewidth=0.9),
                    capprops=dict(color=PALETTE["primary"],
                                   linewidth=0.9),
                    flierprops=dict(marker="o", color=PALETTE["neutral"],
                                    markersize=2.5, alpha=0.5,
                                    markeredgewidth=0))

    # Global mean line
    all_covs = [c for g in data_per_class for c in g]
    global_mean = np.mean(all_covs)
    ax.axhline(global_mean, color=PALETTE["highlight"],
               linewidth=1.3, linestyle="--",
               label=f"Global mean ({global_mean:.1f}%)")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("Coverage (%)")
    ax.set_title("Per-Video Coverage Distribution by Class", pad=8)
    ax.legend(loc="upper right", framealpha=0.85)
    ax.set_xlim(0.3, len(classes) + 0.7)

    fig.tight_layout()
    _save(fig, out_dir, "fig4_coverage_distribution")
    print("[DONE] fig4_coverage_distribution")

# ─── Figure 5: Single-video histogram (real NPZ only) ─────────────────────────

def fig_histogram_sample(json_path, out_dir, npz_base_dir=None,
                          target_class="Arson", target_video=None):
    """
    Plot the event-count histogram for one real video.

    Requires --npz. The chosen record must have a readable .npz on disk;
    records without one are skipped. If none qualifies, the run aborts.
    """
    set_ieee_style()

    if not npz_base_dir:
        raise SystemExit("[ERROR] fig5 needs --npz <base dir of NPZ files>.")

    data = load_json(json_path)

    def _npz_for(rec):
        cls = rec.get("class_name") or _class_from_path(rec.get("npz_path", ""))
        if not cls:
            return None
        p = os.path.join(npz_base_dir, cls, rec["video_name"] + ".npz")
        return p if (os.path.exists(p) and zipfile.is_zipfile(p)) else None

    # Pick the first record of the target class whose NPZ is actually present.
    record, npz_path = None, None
    for r in data:
        cls = r.get("class_name") or _class_from_path(r.get("npz_path", ""))
        if cls != target_class:
            continue
        if target_video and r["video_name"] != target_video:
            continue
        p = _npz_for(r)
        if p:
            record, npz_path = r, p
            break

    if record is None:
        usable = sorted({(r.get("class_name") or "?") for r in data if _npz_for(r)})
        raise SystemExit(
            f"[ERROR] No readable NPZ for class '{target_class}'"
            + (f" / video '{target_video}'" if target_video else "")
            + ".\n        Classes with readable files: "
            + (", ".join(usable) if usable else "none")
            + "\n        Re-run with --hist-class <one of the above>.")

    print(f"[INFO] fig5 using real NPZ: {npz_path}")

    npz_data = np.load(npz_path)
    t = npz_data["t"] / 1e6
    counts, bin_edges = np.histogram(t, bins=200)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    median_val = np.median(counts)
    mad = np.median(np.abs(counts - median_val))
    threshold = median_val + 2 * mad

    _draw_histogram(bin_centers, counts, threshold,
                    record.get("active_windows", []), record, out_dir)

# ─── Gallery: one histogram per readable video, for picking a sample ──────────

def fig_histogram_gallery(json_path, out_dir, npz_base_dir=None,
                          target_class=None, limit=60):
    """
    Render one histogram per readable video into <out_dir>/gallery/, and print
    a table ranked by visual contrast (peak / median event count) so a strong
    example is easy to pick. Pass the winner back as --hist-video.
    """
    set_ieee_style()
    if not npz_base_dir:
        raise SystemExit("[ERROR] gallery needs --npz <base dir of NPZ files>.")

    data = load_json(json_path)
    gdir = os.path.join(out_dir, "gallery")
    os.makedirs(gdir, exist_ok=True)

    rows, made = [], 0
    for r in data:
        cls = r.get("class_name") or _class_from_path(r.get("npz_path", ""))
        if not cls or (target_class and cls != target_class):
            continue
        p = os.path.join(npz_base_dir, cls, r["video_name"] + ".npz")
        if not (os.path.exists(p) and zipfile.is_zipfile(p)):
            continue
        try:
            t = np.load(p)["t"] / 1e6
        except Exception:
            continue
        if t.size == 0 or t.max() <= t.min():
            continue

        counts, edges = np.histogram(t, bins=200)
        centers = 0.5 * (edges[:-1] + edges[1:])
        med = float(np.median(counts))
        mad = float(np.median(np.abs(counts - med)))
        thr = med + 2 * mad
        wins = r.get("active_windows", [])
        contrast = (counts.max() / med) if med > 0 else float("inf")

        _draw_histogram(centers, counts, thr, wins, r, gdir,
                        name=f"{cls}_{r['video_name']}")
        rows.append((cls, r["video_name"], r.get("coverage_percent", 0.0),
                     len(wins), float(t.max() - t.min()), contrast,
                     r.get("result", "?")))
        made += 1
        if made >= limit:
            break

    if not rows:
        raise SystemExit("[ERROR] No readable NPZ files matched.")

    rows.sort(key=lambda x: -x[5])
    print("\n" + "=" * 86)
    print(f"GALLERY — {made} histograms in {gdir}")
    print("Ranked by contrast (peak / median). Higher = clearer bursts.")
    print("=" * 86)
    print(f"{'Class':<14}{'Video':<28}{'Cov%':>7}{'Wins':>6}{'Dur s':>8}"
          f"{'Contrast':>10}{'GT':>9}")
    print("-" * 86)
    for c, v, cov, nw, dur, con, res in rows:
        print(f"{c:<14}{v:<28}{cov:>7.1f}{nw:>6}{dur:>8.1f}{con:>10.1f}{res:>9}")

    top = rows[0]
    print("\nPick one, then render it as the paper figure:")
    print(f"  --only fig5 --hist-class {top[0]} --hist-video {top[1]}")

# ─── Histogram drawing (shared by fig5 and the gallery) ───────────────────────

def _draw_histogram(bin_centers, counts, threshold, active_windows, record, out_dir,
                    name="fig5_histogram_sample"):
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    bar_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

    # Active window shading (behind bars)
    for (ws, we) in active_windows:
        ax.axvspan(ws, we, alpha=0.22, color=PALETTE["highlight"],
                   zorder=1, label="_nolegend_")

    # Bars
    ax.bar(bin_centers, counts, width=bar_width * 0.93,
           color="#7c3aed", alpha=0.78, zorder=2, label="Event count")

    # Threshold line
    ax.axhline(y=threshold, color=PALETTE["accent"],
               linewidth=1.8, linestyle="--", zorder=5,
               label=rf"$\theta_2$ = Median + 2×MAD  ({int(threshold):,})")

    # Coverage annotation
    cov = record.get("coverage_percent", 0)
    ax.text(0.98, 0.95, f"Selected: {cov:.1f}% of video",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9c3",
                      edgecolor="#d97706", linewidth=0.8),
            color="#92400e", fontweight="bold")

    legend_handles = [
        mpatches.Patch(color="#7c3aed", alpha=0.78, label="Event count"),
        plt.Line2D([0], [0], color=PALETTE["accent"], linewidth=1.8,
                   linestyle="--",
                   label=rf"$\theta_2$ = Median + 2×MAD  ({int(threshold):,})"),
        mpatches.Patch(color=PALETTE["highlight"], alpha=0.4,
                       label="Active window"),
    ]
    ax.legend(handles=legend_handles, loc="center right",
            fontsize=7.5, framealpha=0.88, edgecolor="#d1d5db")

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Event Count")
    cls  = record.get("class_name", "")
    vid  = record.get("video_name", "")
    ax.set_title(f"Event Activity Histogram", pad=7)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save(fig, out_dir, name)
    print(f"[DONE] {name}   ({cls}/{vid})")

# ─── Save helper ──────────────────────────────────────────────────────────────

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ─── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate E-VEZA paper figures.")
    parser.add_argument("--json",  required=True,
                        help="Path to global_active_windows.json")
    parser.add_argument("--out",   default="paper_figures",
                        help="Output directory for figures")
    parser.add_argument("--npz",   default=None,
                        help="(Optional) Base dir of NPZ files, e.g. data/UCF-Crime-DVS. "
                             "Required for exact sensitivity curve and histogram.")
    parser.add_argument("--hist-class",  default="Arson",
                        help="Class to use for the sample histogram figure")
    parser.add_argument("--hist-video",  default=None,
                        help="Specific video name (without .npz) for the histogram. "
                             "If omitted, the first matching record is used.")
    parser.add_argument("--only", default=None,
                        help="Generate only one figure: fig1|fig2|fig3|fig4|fig5|gallery")
    parser.add_argument("--gallery-class", default=None,
                        help="Restrict the gallery to one class (default: all)")
    parser.add_argument("--gallery-limit", type=int, default=60,
                        help="Max histograms to render in gallery mode")
    args = parser.parse_args()

    print("[DEBUG] npz =", repr(args.npz), "| isdir =",
          os.path.isdir(args.npz) if args.npz else "N/A")
    print(f"[INFO] Loading {args.json}")
    data   = load_json(args.json)
    groups = group_by_class(data)
    stats  = per_class_stats(groups)

    print(f"[INFO] Found {len(data)} records across {len(groups)} classes.")
    print(f"[INFO] Output directory: {args.out}\n")

    only = args.only

    if only == "gallery":
        fig_histogram_gallery(args.json, args.out,
                              npz_base_dir=args.npz,
                              target_class=args.gallery_class,
                              limit=args.gallery_limit)
        return

    if only in (None, "fig1"): fig_coverage_barplot(stats, args.out)
    if only in (None, "fig2"): fig_threshold_sensitivity(args.json, args.out,
                                   n_sample=10000, npz_base_dir=args.npz)
    if only in (None, "fig3"): fig_duration_vs_coverage(stats, args.out)
    if only in (None, "fig4"): fig_coverage_distribution(groups, args.out)
    if only in (None, "fig5"): fig_histogram_sample(args.json, args.out,
                                   npz_base_dir=args.npz,
                                   target_class=args.hist_class,
                                   target_video=args.hist_video)

    print(f"\n✓ All figures saved to  {args.out}/")

if __name__ == "__main__":
    main()