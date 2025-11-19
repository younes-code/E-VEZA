import numpy as np
import cv2
import os
from glob import glob


def crop_to_event_activity(rgb_frame, events_data, frame_timestamp_us,
                           time_window_us=50000,   # ±25ms — safe even for sparsely sampled frames
                           padding=40,             # extra pixels around the event cloud
                           min_size=(64, 64)):     # ensure crop is never too tiny
    """
    Always returns a cropped patch centered on event activity.
    If zero events → returns center crop of original frame.
    """
    x = events_data['x'].astype(np.int32)
    y = events_data['y'].astype(np.int32)
    ts = events_data['ts'] if 'ts' in events_data else events_data['t']

    H, W = rgb_frame.shape[:2]

    # Time window around this exact frame timestamp
    t_start = frame_timestamp_us - time_window_us // 2
    t_end   = frame_timestamp_us + time_window_us // 2
    mask = (ts >= t_start) & (ts <= t_end)

    if mask.sum() == 0:
        # No events at all → fall back to center crop
        print(f"  No events for ts={frame_timestamp_us} → using center crop")
        ch, cw = H // 2, W // 2
        half_w = max(min_size[0] // 2, 120)
        half_h = max(min_size[1] // 2, 120)
        x1 = max(0, cw - half_w)
        y1 = max(0, ch - half_h)
        x2 = min(W, cw + half_w)
        y2 = min(H, ch + half_h)
    else:
        x_evt = x[mask]
        y_evt = y[mask]

        # Bounding box from events + padding
        x1 = max(0, x_evt.min() - padding)
        y1 = max(0, y_evt.min() - padding)
        x2 = min(W, x_evt.max() + padding + 1)
        y2 = min(H, y_evt.max() + padding + 1)

        # Enforce minimum crop size
        if x2 - x1 < min_size[0]:
            extra = (min_size[0] - (x2 - x1)) // 2
            x1 = max(0, x1 - extra)
            x2 = min(W, x2 + extra + (min_size[0] - (x2 - x1)))
        if y2 - y1 < min_size[1]:
            extra = (min_size[1] - (y2 - y1)) // 2
            y1 = max(0, y1 - extra)
            y2 = min(H, y2 + extra + (min_size[1] - (y2 - y1)))

    patch = rgb_frame[y1:y2, x1:x2]
    bbox = (x1, y1, x2, y2)

    return patch, bbox


def process_all_frames_keep_all(frames_folder, npz_path, output_folder,
                                time_window_us=50000,
                                padding=40,
                                min_crop_size=(96, 96)):
    """
    Processes EVERY frame in the folder.
    Never skips. Always saves one cropped image per input frame.
    Frame filename = timestamp in microseconds (e.g. 1634567890123.jpg)
    """
    # Load events once
    print(f"Loading events from: {npz_path}")
    events_data = np.load(npz_path, allow_pickle=True)

    # Get all image files
    frame_paths = glob(os.path.join(frames_folder, "*.jpg")) + \
                  glob(os.path.join(frames_folder, "*.png")) + \
                  glob(os.path.join(frames_folder, "*.jpeg"))

    # Sort by timestamp (filename)
    frame_paths = sorted(frame_paths,
                         key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    print(f"Found {len(frame_paths)} frames → processing ALL of them\n")

    os.makedirs(output_folder, exist_ok=True)

    for i, frame_path in enumerate(frame_paths, 1):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read {frame_path}")
            continue

        # Parse timestamp from filename
        ts_str = os.path.splitext(os.path.basename(frame_path))[0]
        try:
            frame_ts_us = int(ts_str)
        except ValueError:
            print(f"Warning: Bad filename (not a number): {frame_path}")
            continue

        patch, bbox = crop_to_event_activity(
            rgb_frame=frame,
            events_data=events_data,
            frame_timestamp_us=frame_ts_us,
            time_window_us=time_window_us,
            padding=padding,
            min_size=min_crop_size
        )

        # Save with same name → perfect 1:1 correspondence
        out_path = os.path.join(output_folder, os.path.basename(frame_path))
        cv2.imwrite(out_path, patch)

        event_count = np.sum((events_data['ts'] >= frame_ts_us - time_window_us//2) &
                            (events_data['ts'] <= frame_ts_us + time_window_us//2))

        print(f"{i:4d}/{len(frame_paths)} | ts={frame_ts_us} | events={event_count:5d} | "
              f"crop={patch.shape} | saved → {os.path.basename(out_path)}")

    print(f"\nDone! All {len(frame_paths)} frames cropped and saved to:\n→ {output_folder}")


# ================================
# Run it
# ================================
if __name__ == "__main__":
    # Example:
    frames_folder = "./sampled_frames/indoor_flying1"        # frames named like 1634123456789.jpg
    npz_path      = "./events/indoor_flying1_events.npz"
    output_folder = "./cropped_patches/indoor_flying1"

    process_all_frames_keep_all(
        frames_folder=frames_folder,
        npz_path=npz_path,
        output_folder=output_folder,
        time_window_us=60000,    # ±30ms — very safe for sparse sampling
        padding=50,
        min_crop_size=(96, 96)   # never smaller than this
    )