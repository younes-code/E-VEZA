import os
import numpy as np
import re
from glob import glob

def parse_time(t_str):
    # Convert "mm:ss.s" to seconds
    m, s = re.match(r"(\d+):([\d\.]+)", t_str).groups()
    return int(m) * 60 + float(s)

def load_annotations(txt_path):
    annotations = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            video = parts[0]
            start = parse_time(parts[1])
            end = parse_time(parts[2])
            annotations.setdefault(video, []).append((start, end))
    return annotations

def detect_active_windows(npz_path):
    data = np.load(npz_path)
    t = data["t"] / 1e6  # convert to seconds
    num_bins = 200
    counts, bin_edges = np.histogram(t, bins=num_bins)
    mean_val = counts.mean()
    std_val = counts.std()
    threshold = mean_val + 1 * std_val
    active_bins = counts > threshold

    active_windows = []
    start, end = None, None
    for i, active in enumerate(active_bins):
        if active and start is None:
            start = bin_edges[i]
            end = bin_edges[i+1]
        elif active:
            end = bin_edges[i+1]
        elif not active and start is not None:
            active_windows.append((start, end))
            start, end = None, None
    if start is not None:
        active_windows.append((start, end))
    return active_windows

def check_overlap(active_windows, annotated_windows):
    # return True if ANY overlap exists
    for a_start, a_end in active_windows:
        for b_start, b_end in annotated_windows:
            if max(a_start, b_start) < min(a_end, b_end):
                return True
    return False

def evaluate(npz_dir, annotation_txt):
    annotations = load_annotations(annotation_txt)
    npz_files = glob(os.path.join(npz_dir, "*.npz"))
    total_videos = 0
    total_success = 0
    per_video_results = []

    for npz_path in npz_files:
        video_name = os.path.splitext(os.path.basename(npz_path))[0]
        active_windows = detect_active_windows(npz_path)
        total_videos += 1

        if video_name in annotations:
            success = check_overlap(active_windows, annotations[video_name])
            if success:
                total_success += 1
            per_video_results.append((video_name, success))
        else:
            # no annotation → automatically failure
            per_video_results.append((video_name, False))

    # Print per-video results
    print("Video\tDetected")
    for v, success in per_video_results:
        print(f"{v}\t{'Success' if success else 'Failure'}")

    # Print overall metric
    overall_percent = (total_success / total_videos * 100) if total_videos else 0
    print(f"\nOverall detection accuracy: {total_success}/{total_videos} ({overall_percent:.1f}%)")

if __name__ == "__main__":
    npz_dir = "data/UCF-Crime-DVS/RoadAccidents"  # change as needed
    annotation_txt = "data/uca_annotations/UCFCrime_Train.txt"
    evaluate(npz_dir, annotation_txt)
