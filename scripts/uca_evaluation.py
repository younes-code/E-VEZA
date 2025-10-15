import os
import numpy as np
import re
from glob import glob

def parse_time(t_str):
    try:
        m, s = re.match(r"(\d+):([\d\.]+)", t_str).groups()
        return int(m) * 60 + float(s)
    except Exception as e:
        print(f"[ERROR] Failed to parse time '{t_str}': {e}")
        return None

def load_annotations(txt_path):
    print(f"[INFO] Loading annotations from: {txt_path}")
    annotations = {}
    if not os.path.exists(txt_path):
        print(f"[ERROR] Annotation file not found: {txt_path}")
        return annotations

    with open(txt_path, "r") as f:
        for i, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 3:
                print(f"[WARNING] Skipping malformed line {i}: '{line.strip()}'")
                continue

            video = parts[0]
            start = parse_time(parts[1])
            end = parse_time(parts[2])

            if start is None or end is None:
                print(f"[WARNING] Skipping invalid times in line {i}")
                continue

            annotations.setdefault(video, []).append((start, end))

    print(f"[INFO] Loaded annotations for {len(annotations)} videos.")
    return annotations

def detect_active_windows(npz_path):
    print(f"[INFO] Processing: {os.path.basename(npz_path)}")
    try:
        data = np.load(npz_path)
        if "t" not in data:
            print(f"[ERROR] Missing 't' key in {npz_path}")
            return []

        t = data["t"] / 1e6  # convert microseconds to seconds
        num_bins = 200
        counts, bin_edges = np.histogram(t, bins=num_bins)

        mean_val = counts.mean()
        std_val = counts.std()
        threshold = mean_val + 1 * std_val
        active_bins = counts > threshold

        print(f"[DEBUG] mean={mean_val:.2f}, std={std_val:.2f}, threshold={threshold:.2f}")
        print(f"[DEBUG] Active bins detected: {active_bins.sum()} / {num_bins}")

        active_windows = []
        start, end = None, None
        for i, active in enumerate(active_bins):
            if active and start is None:
                start = bin_edges[i]
                end = bin_edges[i + 1]
            elif active:
                end = bin_edges[i + 1]
            elif not active and start is not None:
                active_windows.append((start, end))
                start, end = None, None

        if start is not None:
            active_windows.append((start, end))

        print(f"[INFO] Detected {len(active_windows)} active window(s)")
        return active_windows

    except Exception as e:
        print(f"[ERROR] Failed to process {npz_path}: {e}")
        return []

def check_overlap(active_windows, annotated_windows):
    for a_start, a_end in active_windows:
        for b_start, b_end in annotated_windows:
            if max(a_start, b_start) < min(a_end, b_end):
                print(f"[DEBUG] Overlap found: active=({a_start:.1f}-{a_end:.1f}), "
                      f"annot=({b_start:.1f}-{b_end:.1f})")
                return True
    return False

def evaluate(npz_dir, annotation_txt):
    print(f"[INFO] Starting evaluation...")
    print(f"[INFO] NPZ directory: {npz_dir}")
    annotations = load_annotations(annotation_txt)

    npz_files = glob(os.path.join(npz_dir, "*.npz"))
    if not npz_files:
        print(f"[WARNING] No .npz files found in {npz_dir}")
        return

    total_videos = 0
    total_success = 0
    per_video_results = []

    for npz_path in npz_files:
        video_name = os.path.splitext(os.path.basename(npz_path))[0]
        print(f"\n[INFO] === Evaluating {video_name} ===")

        active_windows = detect_active_windows(npz_path)
        total_videos += 1

        if video_name in annotations:
            success = check_overlap(active_windows, annotations[video_name])
            total_success += int(success)
            result_text = "Success" if success else "Failure"
        else:
            print(f"[WARNING] No annotation found for {video_name}")
            success = False
            result_text = "No Annotation"

        per_video_results.append((video_name, result_text))
        print(f"[RESULT] {video_name}: {result_text}")

    print("\n=== Summary ===")
    print("Video\tResult")
    for v, result in per_video_results:
        print(f"{v}\t{result}")

    overall_percent = (total_success / total_videos * 100) if total_videos else 0
    print(f"\n[INFO] Overall detection accuracy: {total_success}/{total_videos} ({overall_percent:.1f}%)")

if __name__ == "__main__":
    npz_dir = "data/UCF-Crime-DVS/RoadAccidents"
    annotation_txt = "data/uca_annotations/UCFCrime_Train.txt"
    evaluate(npz_dir, annotation_txt)
