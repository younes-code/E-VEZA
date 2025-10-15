import os
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


# -------------------------
# Utility functions
# -------------------------
def parse_time(t_str):
    m, s = re.match(r"(\d+):([\d\.]+)", t_str).groups()
    return int(m) * 60 + float(s)


def load_annotations(txt_path):
    annotations = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            video, start, end = parts[0], parse_time(parts[1]), parse_time(parts[2])
            annotations.setdefault(video, []).append((start, end))
    print(f"[INFO] Loaded annotations for {len(annotations)} videos.")
    return annotations


# -------------------------
# Core detection logic
# -------------------------
def detect_active_windows(npz_path, visualize=False):
    data = np.load(npz_path)
    if "t" not in data:
        print(f"[ERROR] Missing 't' key in {npz_path}")
        return [], 0, 0

    t = data["t"] / 1e6  # convert to seconds
    counts, bin_edges = np.histogram(t, bins=200)

    # Adaptive threshold (Median + 2*MAD)
    median_val = np.median(counts)
    mad = np.median(np.abs(counts - median_val))
    threshold = median_val + 1 * mad

    active_bins = counts > threshold

    # Merge consecutive bins safely
    active_windows = []
    start, end = None, None
    for i, active in enumerate(active_bins):
        if active and start is None:
            start = bin_edges[i]
            end = bin_edges[i + 1]  # ensure valid end even if single-bin
        elif active:
            end = bin_edges[i + 1]
        elif not active and start is not None:
            active_windows.append((start, end))
            start, end = None, None
    if start is not None and end is not None:
        active_windows.append((start, end))

    # Compute coverage safely
    total_duration = t.max() - t.min()
    if total_duration <= 0:
        return active_windows, 0, 0

    active_duration = sum((e - s) for s, e in active_windows if e is not None)
    coverage_ratio = active_duration / total_duration if total_duration > 0 else 0

    # Visualization
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


# -------------------------
# Evaluation + stats
# -------------------------
def check_overlap(active_windows, annotated_windows):
    for a_start, a_end in active_windows:
        for b_start, b_end in annotated_windows:
            if max(a_start, b_start) < min(a_end, b_end):
                return True
    return False


def evaluate(npz_dir, annotation_txt, visualize=False):
    annotations = load_annotations(annotation_txt)
    npz_files = glob(os.path.join(npz_dir, "*.npz"))
    if not npz_files:
        print(f"[ERROR] No NPZ files found in {npz_dir}")
        return 0, 0  # Return dummy values

    total_videos, total_success = 0, 0
    total_coverage, results = 0, []

    for npz_path in npz_files:
        video_name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\n[INFO] === Evaluating {video_name} ===")
        active_windows, coverage_ratio, total_duration = detect_active_windows(npz_path, visualize)

        total_videos += 1
        total_coverage += coverage_ratio

        if video_name in annotations:
            success = check_overlap(active_windows, annotations[video_name])
            total_success += int(success)
            result = "Success" if success else "Failure"
        else:
            result = "No Annotation"

        results.append((video_name, result, coverage_ratio * 100))
        print(f"[RESULT] {video_name}: {result} | Active coverage = {coverage_ratio*100:.2f}%")

    # Summary for this class
    print("\n=== Summary ===")
    print("Video\tResult\tCoverage (%)")
    for v, r, c in results:
        print(f"{v}\t{r}\t{c:.1f}")

    avg_cov = (total_coverage / total_videos * 100) if total_videos else 0
    overall_acc = (total_success / total_videos * 100) if total_videos else 0

    print(f"\n[INFO] Overall detection accuracy: {overall_acc:.1f}%")
    print(f"[INFO] Average video coverage (frames kept): {avg_cov:.1f}%")
    print(f"[INFO] Estimated optimization (frames skipped): {100 - avg_cov:.1f}%")

    return overall_acc, avg_cov  # Return per-class stats


# -------------------------
if __name__ == "__main__":
    base_dir = "data/UCF-Crime-DVS"
    annotation_txt = "data/uca_annotations/UCFCrime_Train.txt"

    class_dirs = [d for d in glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

    global_acc_list, global_cov_list = [], []

    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        print(f"\n===============================")
        print(f"[INFO] Running evaluation for class: {class_name}")
        print(f"===============================")

        acc, cov = evaluate(class_dir, annotation_txt, visualize=False)
        global_acc_list.append(acc)
        global_cov_list.append(cov)

    # ---- Global summary ----
    if global_acc_list:
        overall_acc = np.mean(global_acc_list)
        overall_cov = np.mean(global_cov_list)
        print("\n#################################")
        print("######## GLOBAL SUMMARY #########")
        print("#################################")
        print(f"[INFO] Mean accuracy across classes: {overall_acc:.2f}%")
        print(f"[INFO] Mean coverage across classes: {overall_cov:.2f}%")
        print(f"[INFO] Mean optimization (frames skipped): {100 - overall_cov:.2f}%")
