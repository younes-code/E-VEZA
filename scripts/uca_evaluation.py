import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


# -------------------------
# Utility functions
# -------------------------
def parse_time(t_str):
    try:
        m, s = re.match(r"(\d+):([\d\.]+)", t_str).groups()
        return int(m) * 60 + float(s)
    except Exception as e:
        print(f"[ERROR] Failed to parse time '{t_str}': {e}")
        return None


def load_annotations(txt_path):
    annotations = {}
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


# -------------------------
# Core detection logic
# -------------------------
import time
import zipfile

def detect_active_windows(npz_path, visualize=False, timeout=10):
    """Detect active temporal windows from DVS event timestamps."""
    start_time = time.time()
    try:
        # Safety: skip corrupted or non-zip files
        if not zipfile.is_zipfile(npz_path):
            raise zipfile.BadZipFile("Not a valid npz file")

        data = np.load(npz_path)

        if "t" not in data:
            print(f"[ERROR] Missing 't' key in {npz_path}")
            return [], 0, 0

        t = data["t"] / 1e6  # convert µs → seconds
        counts, bin_edges = np.histogram(t, bins=200)

        # Adaptive threshold (Median + 2×MAD)
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
        if total_duration <= 0:
            return active_windows, 0, 0
        active_duration = sum((e - s) for s, e in active_windows if e is not None)
        coverage_ratio = active_duration / total_duration

        # Optional visualization
        if visualize:
            plt.figure(figsize=(10, 4))
            plt.plot(bin_edges[:-1], counts, color="purple")
            plt.axhline(threshold, color="orange", linestyle="--", label="Threshold")
            for (s, e) in active_windows:
                if e is not None:
                    plt.axvspan(s, e, color="green", alpha=0.3)
            plt.title(f"Activity in {os.path.basename(npz_path)}")
            plt.xlabel("Time (s)")
            plt.ylabel("Event count")
            plt.legend()
            plt.show()

        return active_windows, coverage_ratio, total_duration

    except Exception as e:
        # Log bad files
        with open("bad_files.log", "a") as logf:
            logf.write(f"{npz_path}: {e}\n")
        print(f"[SKIP] {npz_path} ({e})")
        return [], 0, 0



# -------------------------
# Evaluation helpers
# -------------------------
def check_overlap(active_windows, annotated_windows):
    for a_start, a_end in active_windows:
        for b_start, b_end in annotated_windows:
            if max(a_start, b_start) < min(a_end, b_end):
                return True
    return False


def evaluate_class(class_dir, annotations, output_root, visualize=False):
    """Run evaluation for one class folder."""
    class_name = os.path.basename(class_dir)
    npz_files = [
        f for f in glob(os.path.join(class_dir, "*.npz"))
        if os.path.getsize(f) > 0 and zipfile.is_zipfile(f)
    ]
    if not npz_files:
        print(f"[WARNING] No NPZ files found in {class_dir}")

        return None

    total_videos, total_success, total_coverage = 0, 0, 0
    results = []

    for npz_path in npz_files:
        video_name = os.path.splitext(os.path.basename(npz_path))[0]
        active_windows, coverage_ratio, total_duration = detect_active_windows(npz_path, visualize)

        total_videos += 1
        total_coverage += coverage_ratio

        if video_name in annotations:
            success = check_overlap(active_windows, annotations[video_name])
            total_success += int(success)
            result = "Success" if success else "Failure"
        else:
            result = "No Annotation"

        results.append({
            "video_name": video_name,
            "result": result,
            "coverage_percent": round(coverage_ratio * 100, 2),
            "total_duration_sec": round(total_duration, 2),
            "active_windows": [(float(s), float(e)) for s, e in active_windows],
        })

    avg_cov = (total_coverage / total_videos * 100) if total_videos else 0
    overall_acc = (total_success / total_videos * 100) if total_videos else 0
    optimization = 100 - avg_cov

    print(f"\n=== {class_name} Summary ===")
    print(f"[INFO] Accuracy: {overall_acc:.1f}% | Coverage: {avg_cov:.1f}% | Optimization: {optimization:.1f}%")

    # Save per-class results
    os.makedirs(output_root, exist_ok=True)
    results_csv = os.path.join(output_root, f"{class_name}_results.csv")
    summary_csv = os.path.join(output_root, f"{class_name}_summary.csv")

    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "result", "coverage_percent", "total_duration_sec", "active_windows"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Class", class_name])
        writer.writerow(["Total Videos", total_videos])
        writer.writerow(["Accuracy (%)", round(overall_acc, 2)])
        writer.writerow(["Average Coverage (%)", round(avg_cov, 2)])
        writer.writerow(["Optimization (%)", round(optimization, 2)])

    return {
        "class": class_name,
        "accuracy": overall_acc,
        "coverage": avg_cov,
        "optimization": optimization,
        "videos": total_videos,
    }

    return overall_acc, avg_cov  # Return per-class stats

def evaluate_all(base_dir, annotation_txt, output_root="class_results", visualize=False):
    """Run evaluation on all class folders and save global summary."""
    annotations = load_annotations(annotation_txt)
    class_dirs = [d for d in glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    print(f"[INFO] Found {len(class_dirs)} class folders.")

    global_summary = []

    for class_dir in class_dirs:
        summary = evaluate_class(class_dir, annotations, output_root, visualize)
        if summary:
            global_summary.append(summary)

    # Save global summary
    global_csv = os.path.join(output_root, "evaluation_global_summary.csv")
    with open(global_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Accuracy (%)", "Coverage (%)", "Optimization (%)", "Videos"])
        for s in global_summary:
            writer.writerow([s["class"], round(s["accuracy"], 2), round(s["coverage"], 2), round(s["optimization"], 2), s["videos"]])

    print(f"\n[INFO] Global summary saved to: {global_csv}")
    
if __name__ == "__main__":
    base_dir = "data/UCF-Crime-DVS"
    annotation_txt = "data/uca_annotations/UCFCrime_Train.txt"
    evaluate_all(base_dir, annotation_txt, output_root="class_results", visualize=False)
