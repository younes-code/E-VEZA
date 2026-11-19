
import os
import re
import json
import time
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from glob import glob


# ==========================
# Utility Functions
# ==========================
def parse_time(t_str):
    """Parse annotation time 'M:SS.ss' → seconds."""
    try:
        m, s = re.match(r"(\d+):([\d\.]+)", t_str).groups()
        return int(m) * 60 + float(s)
    except Exception as e:
        print(f"[ERROR] Failed to parse time '{t_str}': {e}")
        return None


def load_annotations(txt_path):
    """Load annotation file into a dict: {video_name: [(start, end), ...]}"""
    annotations = {}
    if not os.path.exists(txt_path):
        print(f"[WARNING] No annotation file found at {txt_path}")
        return annotations

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            video, start, end = parts[0], parse_time(parts[1]), parse_time(parts[2])
            if start is None or end is None:
                continue
            annotations.setdefault(video, []).append((start, end))

    print(f"[INFO] Loaded annotations for {len(annotations)} videos.")
    return annotations


# ==========================
# Histogram — Paper-Quality Figure
# ==========================
def save_histogram_paper(t, npz_path, output_dir, annotated_windows=None, active_windows=None,
                         coverage_ratio=None, class_name=None):
    """
    Generate a paper-quality histogram figure showing:
    - Event count bars
    - MAD threshold line
    - Active windows shading
    - Coverage % annotation
    """
    counts, bin_edges = np.histogram(t, bins=200)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Compute threshold
    median_val = np.median(counts)
    mad = np.median(np.abs(counts - median_val))
    threshold = median_val + 2 * mad

    # Get video base name
    video_base = os.path.splitext(os.path.basename(npz_path))[0]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Active windows shading
    if active_windows:
        for (ws, we) in active_windows:
            ax.axvspan(ws, we, alpha=0.25, color="#f59e0b", zorder=1, label="_nolegend_")

    # Event count bars
    bar_width = bin_edges[1] - bin_edges[0]
    ax.bar(bin_centers, counts, width=bar_width * 0.95,
           color="#9333ea", alpha=0.75, zorder=2, label="Event count")

    # Threshold line
    ax.axhline(y=threshold, color="#dc2626", linewidth=1.8,
               linestyle="--", zorder=5,
               label=f"T = Median + 2×MAD ({int(threshold):,})")

    # Coverage box
    if coverage_ratio is not None:
        coverage_text = f"Selected: {coverage_ratio * 100:.1f}% of video"
        ax.text(0.98, 0.96, coverage_text,
                transform=ax.transAxes,
                fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fef9c3",
                          edgecolor="#d97706", linewidth=1),
                color="#92400e", fontweight="bold")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Event Count", fontsize=11)
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=t.min(), right=t.max())

    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#d1d5db")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color="#9333ea", alpha=0.75, label="Event count"),
        plt.Line2D([0], [0], color="#dc2626", linewidth=1.8,
                   linestyle="--", label=f"T = Median + 2×MAD ({int(threshold):,})"),
        mpatches.Patch(color="#f59e0b", alpha=0.5, label="Active window"),
    ]

    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=8.5, framealpha=0.9, edgecolor="#d1d5db")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{video_base}_histogram_paper.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[PLOT SAVED] {out_path}")
    return out_path


# ==========================
# Core Detection Logic
# ==========================
def detect_active_windows(npz_path, timeout=10):
    """Detect active time intervals from DVS event timestamps."""
    start_time = time.time()
    try:
        if not zipfile.is_zipfile(npz_path):
            raise zipfile.BadZipFile("Not a valid npz file")

        data = np.load(npz_path)
        if "t" not in data:
            raise KeyError("Missing key 't' in npz file")

        t = data["t"] / 1e6  # µs → seconds
        counts, bin_edges = np.histogram(t, bins=200)

        median_val = np.median(counts)
        mad = np.median(np.abs(counts - median_val))
        threshold = median_val + 2 * mad

        active_bins = counts > threshold
        active_windows = []
        start, end = None, None

        for i, active in enumerate(active_bins):
            if time.time() - start_time > timeout:
                raise TimeoutError("Processing timeout exceeded")
            if active and start is None:
                start = bin_edges[i]
                end = bin_edges[i + 1]
            elif active:
                end = bin_edges[i + 1]
            elif not active and start is not None:
                active_windows.append((start, end))
                start, end = None, None

        if start is not None and end is not None:
            active_windows.append((start, end))

        total_duration = t.max() - t.min()
        active_duration = sum((e - s) for s, e in active_windows)
        coverage_ratio = active_duration / total_duration if total_duration > 0 else 0

        return t, counts, threshold, active_windows, coverage_ratio, total_duration

    except Exception as e:
        with open("bad_files.log", "a") as logf:
            logf.write(f"{npz_path}: {e}\n")
        print(f"[SKIP] {npz_path} ({e})")
        return None, None, None, [], 0, 0


# ==========================
# Evaluation Helpers
# ==========================
def check_overlap(active_windows, annotated_windows):
    for a_start, a_end in active_windows:
        for b_start, b_end in annotated_windows:
            if max(a_start, b_start) < min(a_end, b_end):
                return True
    return False

# ==========================
# Ablation: Multiple Threshold Variants
# ==========================
def detect_active_windows_variant(npz_path, variant="mad", timeout=10):
    """
    Same as detect_active_windows but with swappable threshold.
    
    variant options:
        "mad"    → Median + 2×MAD          (your current method)
        "mean"   → Mean + 1×Std            (E-VEZA-Mean)
        "fixed"  → 80th percentile         (E-VEZA-Fixed)
        "median" → Median only             (E-VEZA-Median)
    """
    start_time = time.time()
    try:
        if not zipfile.is_zipfile(npz_path):
            raise zipfile.BadZipFile("Not a valid npz file")
        data = np.load(npz_path)
        if "t" not in data:
            raise KeyError("Missing key 't' in npz file")

        t = data["t"] / 1e6
        counts, bin_edges = np.histogram(t, bins=200)

        # ── Threshold selection ──────────────────────────────────────
        if variant == "mad":
            median_val = np.median(counts)
            mad = np.median(np.abs(counts - median_val))
            threshold = median_val + 2 * mad

        elif variant == "mean":
            threshold = counts.mean() + 1.0 * counts.std()

        elif variant == "fixed":
            threshold = np.percentile(counts, 80)

        elif variant == "median":
            threshold = np.median(counts)

        else:
            raise ValueError(f"Unknown variant: {variant}")
        # ─────────────────────────────────────────────────────────────

        active_bins = counts > threshold
        active_windows = []
        start, end = None, None

        for i, active in enumerate(active_bins):
            if time.time() - start_time > timeout:
                raise TimeoutError("Processing timeout exceeded")
            if active and start is None:
                start = bin_edges[i]; end = bin_edges[i + 1]
            elif active:
                end = bin_edges[i + 1]
            elif not active and start is not None:
                active_windows.append((start, end)); start, end = None, None

        if start is not None and end is not None:
            active_windows.append((start, end))

        total_duration = t.max() - t.min()
        active_duration = sum((e - s) for s, e in active_windows)
        coverage_ratio = active_duration / total_duration if total_duration > 0 else 0

        return coverage_ratio, total_duration

    except Exception as e:
        with open("bad_files.log", "a") as logf:
            logf.write(f"{npz_path}: {e}\n")
        return 0, 0


def run_ablation(base_dir, output_path="ablation_results.json"):
    """
    Run all four threshold variants across all NPZ files.
    Saves per-video results and prints the summary table.
    """
    variants = ["mad", "mean", "fixed", "median"]
    npz_files = [
        f for f in glob(os.path.join(base_dir, "*", "*.npz"))
        if os.path.getsize(f) > 0 and zipfile.is_zipfile(f)
    ]

    print(f"[INFO] Running ablation on {len(npz_files)} files...")

    results = {v: [] for v in variants}

    for npz_path in npz_files:
        for variant in variants:
            cov, dur = detect_active_windows_variant(npz_path, variant=variant)
            if dur > 0:
                results[variant].append(cov * 100)

    summary = {}
    for v in variants:
        covs = np.array(results[v])
        summary[v] = {
            "mean_coverage": float(np.mean(covs)),
            "std_coverage":  float(np.std(covs)),
            "n_videos":      len(covs)
        }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== ABLATION RESULTS ===")
    print(f"{'Variant':<20} {'Avg Cov (%)':>12} {'Std Cov (%)':>12}")
    print("-" * 46)
    for v, s in summary.items():
        print(f"{v:<20} {s['mean_coverage']:>12.2f} {s['std_coverage']:>12.2f}")

    return summary


# ==========================
# Main Processing
# ==========================
def process_video(npz_path, annotations, output_root, class_name=None, save_plots=True):
    """Detect active windows for one video and return formatted result."""
    video_name = os.path.splitext(os.path.basename(npz_path))[0]

    t, counts, threshold, active_windows, coverage, total_duration = detect_active_windows(npz_path)

    if t is None:
        return None

    annotated_windows = annotations.get(video_name, None)

    if save_plots:
        hist_dir = os.path.join(output_root, "histograms", class_name if class_name else "unknown")
        save_histogram_paper(
            t=t,
            npz_path=npz_path,
            output_dir=hist_dir,
            annotated_windows=annotated_windows,
            active_windows=active_windows,
            coverage_ratio=coverage,
            class_name=class_name
        )

    if annotated_windows:
        success = check_overlap(active_windows, annotated_windows)
        result = "Success" if success else "Failure"
    else:
        result = "Unknown"

    return {
        "video_name": video_name,
        "class_name": class_name,
        "result": result,
        "coverage_percent": round(coverage * 100, 2),
        "total_duration_sec": round(total_duration, 2),
        "active_windows": [(float(s), float(e)) for s, e in active_windows],
        "threshold": float(threshold) if threshold is not None else None,
    }


def process_class(class_dir, annotations, output_root, save_plots=True):
    """Process all NPZ videos inside one class folder."""
    class_name = os.path.basename(class_dir)
    npz_files = [
        f for f in glob(os.path.join(class_dir, "*.npz"))
        if os.path.getsize(f) > 0 and zipfile.is_zipfile(f)
    ]

    if not npz_files:
        print(f"[WARNING] No NPZ files found in {class_dir}")
        return None

    print(f"[INFO] Processing class: {class_name} ({len(npz_files)} files)")
    results = []

    for npz_path in npz_files:
        result = process_video(npz_path, annotations, output_root,
                               class_name=class_name, save_plots=save_plots)
        if result:
            results.append(result)
            print(f"[OK] {result['video_name']} | "
                  f"coverage={result['coverage_percent']}% | "
                  f"windows={len(result['active_windows'])} | "
                  f"result={result['result']}")

    os.makedirs(output_root, exist_ok=True)
    class_json_path = os.path.join(output_root, f"{class_name}_active_windows.json")
    with open(class_json_path, "w") as jf:
        json.dump(results, jf, indent=2)

    print(f"[SAVED] {class_name} → {class_json_path}")
    return results


def process_all(base_dir, annotation_txt, output_root="outputs/windows",
                save_plots=True, target_class=None):
    """
    Run processing for all class folders OR only one specific class.
    
    Example:
        process_all(base_dir, annotation_txt, target_class="Arson")
    """
    annotations = load_annotations(annotation_txt)

    if target_class is not None:
        class_dir = os.path.join(base_dir, target_class)
        if not os.path.isdir(class_dir):
            print(f"[ERROR] Class folder not found: {class_dir}")
            return []
        class_dirs = [class_dir]
        print(f"[INFO] Running ONLY on class: {target_class}\n")
    else:
        class_dirs = [d for d in glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
        print(f"[INFO] Found {len(class_dirs)} class folders.\n")

    all_results = []
    for class_dir in class_dirs:
        class_results = process_class(class_dir, annotations, output_root,
                                      save_plots=save_plots)
        if class_results:
            all_results.extend(class_results)

    global_json = os.path.join(output_root, "global_active_windows.json")
    with open(global_json, "w") as jf:
        json.dump(all_results, jf, indent=2)

    print(f"\n[INFO] Global results saved → {global_json}")
    return all_results


# ==========================
# Quick single-file test
# ==========================
def plot_single(npz_path, annotation_txt=None, class_name=None, output_dir="paper_figures"):
    annotations = load_annotations(annotation_txt) if annotation_txt else {}
    t, counts, threshold, active_windows, coverage, total_duration = detect_active_windows(npz_path)
    if t is None:
        print("[ERROR] Could not load file.")
        return

    video_name = os.path.splitext(os.path.basename(npz_path))[0]
    annotated_windows = annotations.get(video_name, None)

    save_histogram_paper(
        t=t,
        npz_path=npz_path,
        output_dir=output_dir,
        annotated_windows=annotated_windows,
        active_windows=active_windows,
        coverage_ratio=coverage,
        class_name=class_name
    )
    print(f"\nCoverage: {coverage*100:.2f}%")
    print(f"Active windows: {active_windows}")
    print(f"Threshold: {threshold:,.0f}")
    if annotated_windows:
        overlap = check_overlap(active_windows, annotated_windows)
        print(f"Overlap with GT: {'YES' if overlap else 'NO'}")


# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    base_dir = "data/UCF-Crime-DVS"
    annotation_txt = "data/uca_annotations/UCFCrime_Train.txt"

    # CHANGE THIS to the class you want
    target_class = None  

    process_all(
        base_dir,
        annotation_txt,
        output_root="outputs/windows",
        save_plots=True,
        target_class=target_class
    )


# # ==========================
# # Ablation studies
# # ==========================
# if __name__ == "__main__":
#     base_dir = "data/UCF-Crime-DVS"
#     annotation_txt = "data/uca_annotations/UCFCrime_Train.txt"

#     # Run ablation
#     ablation = run_ablation(base_dir, output_path="ablation_results.json")

