import os
import re
import csv
import json
import time
import zipfile
import numpy as np
import matplotlib.pyplot as plt
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
# Histogram Saving
# ==========================
def save_histogram(t, npz_path, output_dir):
    """Generate and save histogram of event timestamps."""
    plt.figure(figsize=(10, 4))
    plt.hist(t, bins=200)
    plt.title(f"Histogram for {os.path.basename(npz_path)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Event count")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{os.path.basename(npz_path)}_hist.png")
    plt.savefig(out_path)
    plt.close()

    print(f"[PLOT SAVED] {out_path}")


# ==========================
# Core Detection Logic
# ==========================
def detect_active_windows(npz_path, visualize=False, timeout=10):
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

        # Robust thresholding: Median + 2×MAD
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
        active_duration = sum((e - s) for s, e in active_windows if e is not None)
        coverage_ratio = active_duration / total_duration if total_duration > 0 else 0

        return active_windows, coverage_ratio, total_duration, t

    except Exception as e:
        with open("bad_files.log", "a") as logf:
            logf.write(f"{npz_path}: {e}\n")
        print(f"[SKIP] {npz_path} ({e})")
        return [], 0, 0, None


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
# Main Processing
# ==========================
def process_video(npz_path, annotations, output_dir, visualize=False):
    """Detect active windows for one video and optionally evaluate."""
    video_name = os.path.splitext(os.path.basename(npz_path))[0]
    print(f"    → Processing video: {video_name}")

    active_windows, coverage, total_duration, t = detect_active_windows(npz_path, visualize)

    # ---- NEW: histogram saving ----
    if t is not None:
        hist_dir = os.path.join(output_dir, "histograms")
        save_histogram(t, npz_path, hist_dir)
    # ------------------------------

    # Optional evaluation
    success = None
    if video_name in annotations:
        success = check_overlap(active_windows, annotations[video_name])

    # Prepare save data
    result_data = {
        "video_name": video_name,
        "npz_path": npz_path,
        "active_windows": [(float(s), float(e)) for s, e in active_windows],
        "coverage_percent": round(coverage * 100, 2),
        "total_duration_sec": round(total_duration, 2),
        "evaluation_success": bool(success) if success is not None else None,
    }

    os.makedirs(output_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(output_dir, f"{video_name}_windows.json")
    with open(json_path, "w") as jf:
        json.dump(result_data, jf, indent=2)

    # Save TXT
    txt_path = os.path.join(output_dir, f"{video_name}_windows.txt")
    with open(txt_path, "w") as tf:
        for s, e in active_windows:
            tf.write(f"{s:.2f}\t{e:.2f}\n")

    print(f"    [SAVED] {video_name}: {len(active_windows)} intervals")

    return result_data


def process_class(class_dir, annotations, output_root, visualize=False):
    """Process all videos in one class folder."""
    class_name = os.path.basename(class_dir)
    npz_files = [
        f for f in glob(os.path.join(class_dir, "*.npz"))
        if os.path.getsize(f) > 0 and zipfile.is_zipfile(f)
    ]

    print(f"\n[CLASS] Processing '{class_name}' ({len(npz_files)} videos)...")

    if not npz_files:
        print(f"[WARNING] No NPZ files found in {class_dir}")
        return None

    class_output_dir = os.path.join(output_root, class_name)
    all_results = []

    for npz_path in npz_files:
        result = process_video(npz_path, annotations, class_output_dir, visualize)
        all_results.append(result)

    # Save combined JSON for class
    combined_json = os.path.join(output_root, f"{class_name}_all_windows.json")
    with open(combined_json, "w") as jf:
        json.dump(all_results, jf, indent=2)

    print(f"[DONE] Class '{class_name}' finished → saved {class_name}_all_windows.json")
    return all_results


def process_all(base_dir, annotation_txt, output_root="temporal_selections", visualize=False):
    """Run selection + evaluation for all class folders."""
    annotations = load_annotations(annotation_txt)
    class_dirs = [d for d in glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

    print(f"[INFO] Found {len(class_dirs)} class folders.\n")

    os.makedirs(output_root, exist_ok=True)
    summary = []

    for class_dir in class_dirs:
        class_results = process_class(class_dir, annotations, output_root, visualize)
        if class_results:
            summary.extend(class_results)

    global_json = os.path.join(output_root, "global_temporal_selection.json")
    with open(global_json, "w") as jf:
        json.dump(summary, jf, indent=2)

    print(f"\n[GLOBAL] Done. Results saved → {global_json}")
    return summary


# ==========================
# Entry Point
# ==========================
if __name__ == "__main__":
    base_dir = "data/UCF-Crime-DVS"
    annotation_txt = "data/uca_annotations/UCFCrime_Train.txt"
    process_all(base_dir, annotation_txt, output_root="temporal_selections", visualize=False)
