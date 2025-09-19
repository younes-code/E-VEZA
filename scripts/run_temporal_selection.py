import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

def extract_active_windows(npz_path, rgb_dir, output_root, plots_dir):
    data = np.load(npz_path)
    t = data["t"] / 1e6  # convert to seconds

    num_bins = 200
    counts, bin_edges = np.histogram(t, bins=num_bins)
    mean_val = counts.mean()
    std_val = counts.std()
    threshold = mean_val + 2 * std_val
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

    # --- Plot and save histogram with active windows ---
    video_name = os.path.splitext(os.path.basename(npz_path))[0]
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.hist(t, bins=num_bins, color="purple", alpha=0.7)
    plt.title(f"Event Activity: {video_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Event Count")
    min_sec = int(t.min())
    max_sec = int(t.max()) + 1
    plt.xticks(np.arange(min_sec, max_sec, step=1), fontsize=8)
    plt.yticks(fontsize=8)
    for (s, e) in active_windows:
        plt.axvspan(s, e, color="orange", alpha=0.3, label="Active window" if s == active_windows[0][0] else None)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(loc="upper right")
    plot_path = os.path.join(plots_dir, f"{video_name}_event_histogram_with_windows.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # --- Video extraction as before ---
    rgb_video_path = os.path.join(rgb_dir, f"{video_name}.mp4")
    if not os.path.exists(rgb_video_path):
        print(f"RGB video not found: {rgb_video_path}")
        return

    output_dir = os.path.join(output_root, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(rgb_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Processing {video_name} (FPS={fps}, frames={total_frames})")

    for idx, (start_sec, end_sec) in enumerate(active_windows):
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        out_path = os.path.join(output_dir, f"window{idx+1}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        print(f"  Saving window {idx+1}: {start_sec:.2f}s → {end_sec:.2f}s ({start_frame}-{end_frame}) to {out_path}")

        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()

    cap.release()

def batch_process(npz_dir, rgb_dir, output_root, plots_dir):
    npz_files = glob(os.path.join(npz_dir, "*.npz"))
    print(f"Found {len(npz_files)} npz files in {npz_dir}")
    for npz_path in npz_files:
        extract_active_windows(npz_path, rgb_dir, output_root, plots_dir)
    print("Batch processing complete.")

if __name__ == "__main__":
    npz_dir = "data/UCF-Crime-DVS/RoadAccidents"  # change as needed
    rgb_dir = "data/UCF-Crime-RGB/RoadAccidents"  # change as needed
    output_root = os.path.join(os.getcwd(), "active_rgb_windows")
    plots_dir = os.path.join(os.getcwd(), "plots")
    batch_process(npz_dir, rgb_dir, output_root, plots_dir)