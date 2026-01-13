import numpy as np
import cv2
import os
import json
from glob import glob
from sklearn.cluster import DBSCAN  # optional dense cluster focusing


def crop_to_event_activity(rgb_frame, events_data, frame_timestamp_us,
                           time_window_us=50000,
                           padding=40,
                           min_size=(64, 64),
                           min_event_density=5.0,  # Events/ms threshold for anomalies
                           use_dense_cluster=False):  # Focus on densest cluster only?
    """
    Enhanced cropping: Only crops if event density exceeds anomaly threshold.
    Falls back to center if low density or no events.
    Optional: Crops ONLY the densest event cluster (for multi-region focus).
    """
    x = events_data['x'].astype(np.int32)
    y = events_data['y'].astype(np.int32)
    ts = events_data['ts'] if 'ts' in events_data else events_data['t']

    H, W = rgb_frame.shape[:2]

    t_start = frame_timestamp_us - time_window_us // 2
    t_end   = frame_timestamp_us + time_window_us // 2
    mask = (ts >= t_start) & (ts <= t_end)

    event_count = mask.sum()
    window_ms = time_window_us / 1000.0
    density = event_count / window_ms if window_ms > 0 and event_count > 0 else 0

    if density < min_event_density:
        # Low density → likely normal, not anomalous → center crop
        print(f"  [crop_to_event_activity] Low density ({density:.2f} < {min_event_density}) at ts={frame_timestamp_us} → center crop")
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

        if use_dense_cluster and len(x_evt) > 0:
            # Cluster events → take bbox of the DENSEST cluster (highest point count)
            print(f"  [CLUSTER] Clustering {len(x_evt)} events for densest region")
            coords = np.column_stack((x_evt, y_evt))
            db = DBSCAN(eps=20, min_samples=10).fit(coords)  # Tune eps/min_samples for your res
            labels = db.labels_
            unique_labels = set(labels) - {-1}  # Ignore noise
            if unique_labels:
                cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels]
                densest_label = max(cluster_sizes, key=lambda item: item[1])[0]
                dense_mask = labels == densest_label
                x_evt = x_evt[dense_mask]
                y_evt = y_evt[dense_mask]
            else:
                print("  [CLUSTER] No dense clusters found → using all")

        # Bounding box from (dense) events + padding
        x1 = max(0, x_evt.min() - padding)
        y1 = max(0, y_evt.min() - padding)
        x2 = min(W, x_evt.max() + padding + 1)
        y2 = min(H, y_evt.max() + padding + 1)

        # Enforce minimum size
        if x2 - x1 < min_size[0]:
            extra = (min_size[0] - (x2 - x1) + 1) // 2
            x1 = max(0, x1 - extra)
            x2 = min(W, x2 + extra)
        if y2 - y1 < min_size[1]:
            extra = (min_size[1] - (y2 - y1) + 1) // 2
            y1 = max(0, y1 - extra)
            y2 = min(H, y2 + extra)

    patch = rgb_frame[y1:y2, x1:x2]
    bbox = (x1, y1, x2 - x1, y2 - y1)  # (x, y, w, h) format
    return patch, bbox


def crop_and_resize_to_target(rgb_frame, events_data, frame_timestamp_us,
                              time_window_us=50000,
                              padding=40,
                              min_size=(96, 96),
                              target_size="original",
                              min_event_density=5.0,
                              use_dense_cluster=False):
    """
    Main function: Crops around high-density (anomalous) events → resizes.
    """
    patch, bbox = crop_to_event_activity(
        rgb_frame=rgb_frame,
        events_data=events_data,
        frame_timestamp_us=frame_timestamp_us,
        time_window_us=time_window_us,
        padding=padding,
        min_size=min_size,
        min_event_density=min_event_density,
        use_dense_cluster=use_dense_cluster
    )

    H_orig, W_orig = rgb_frame.shape[:2]

    if target_size == "original" or target_size is None:
        target_w, target_h = W_orig, H_orig
    else:
        target_w, target_h = target_size

    patch_resized = cv2.resize(patch, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return patch_resized, bbox


def process_all_frames_resize_back(frames_folder, npz_path, output_folder,
                                   time_window_us=60000,
                                   padding=50,
                                   min_crop_size=(96, 96),
                                   target_size="original",
                                   save_crop_info=False,
                                   active_json_path=None,
                                   min_event_density=5.0,  # Tune for anomalies
                                   use_dense_cluster=False):  # For tighter anomaly focus
    """
    Processes ONLY active frames → crops high-density regions → resizes.
    """
    print(f"Loading events from: {npz_path}")
    events = np.load(npz_path, allow_pickle=True)

    # Load active windows from Script 1's JSON (coherence step)
    active_windows = []
    video_name = os.path.splitext(os.path.basename(npz_path))[0]
    if active_json_path:
        with open(active_json_path, 'r') as f:
            results = json.load(f)
        for res in results:
            if res['video_name'] == video_name:
                active_windows = res['active_windows']
                print(f"  [COHERENCE] Loaded {len(active_windows)} active windows for {video_name}")
                break
        if not active_windows:
            print(f"  [WARNING] No active windows found for {video_name} in {active_json_path}")

    frame_paths = sorted(
        glob(os.path.join(frames_folder, "*.jpg")) +
        glob(os.path.join(frames_folder, "*.png")) +
        glob(os.path.join(frames_folder, "*.jpeg")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
    )

    print(f"Found {len(frame_paths)} frames → filtering to active + high-density cropping/resizing\n")
    os.makedirs(output_folder, exist_ok=True)

    if save_crop_info:
        bbox_list = []

    processed_count = 0
    for i, frame_path in enumerate(frame_paths, 1):
        ts_str = os.path.splitext(os.path.basename(frame_path))[0]
        try:
            frame_ts_us = int(ts_str)
        except ValueError:
            print(f"  [Warning] Bad timestamp: {frame_path}")
            continue

        # Coherence check: Skip if not in global active window
        if active_windows and not is_frame_active(frame_ts_us, active_windows):
            print(f"  [SKIP] Frame ts={frame_ts_us} not in active windows")
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"  [Warning] Could not read: {frame_path}")
            continue

        patch_resized, bbox = crop_and_resize_to_target(
            rgb_frame=frame,
            events_data=events,
            frame_timestamp_us=frame_ts_us,
            time_window_us=time_window_us,
            padding=padding,
            min_size=min_crop_size,
            target_size=target_size,
            min_event_density=min_event_density,
            use_dense_cluster=use_dense_cluster
        )

        # Save resized full-size image
        out_path = os.path.join(output_folder, os.path.basename(frame_path))
        cv2.imwrite(out_path, patch_resized)

        # Count events in window for logging
        t_start = frame_ts_us - time_window_us // 2
        t_end   = frame_ts_us + time_window_us // 2
        ts = events['ts'] if 'ts' in events else events['t']
        event_count = np.sum((ts >= t_start) & (ts <= t_end))
        density = event_count / (time_window_us / 1000.0)

        print(f"{i:4d}/{len(frame_paths)} | ts={frame_ts_us} | events={event_count:5d} | density={density:.2f} | "
              f"output={patch_resized.shape} | {os.path.basename(out_path)}")

        if save_crop_info:
            bbox_list.append({
                "filename": os.path.basename(frame_path),
                "timestamp_us": frame_ts_us,
                "bbox": bbox,
                "event_count": int(event_count),
                "density": density
            })

        processed_count += 1

    if save_crop_info:
        json_path = os.path.join(output_folder, "crop_bboxes.json")
        with open(json_path, 'w') as f:
            json.dump(bbox_list, f, indent=2)
        print(f"\nBounding boxes saved to: {json_path}")

    print(f"\nDONE! Processed {processed_count} anomalous-focused frames (out of {len(frame_paths)}) to:\n→ {output_folder}")


def is_frame_active(frame_ts_us, active_windows):
    """Check if frame timestamp (in us) falls in any active window (in seconds)."""
    frame_ts_sec = frame_ts_us / 1_000_000.0
    for start_sec, end_sec in active_windows:
        if start_sec <= frame_ts_sec <= end_sec:
            return True
    return False


# ================================
# Run it
# ================================
if __name__ == "__main__":
    frames_folder = "data/frames/Fighting017_x264_active"         
    npz_path      = "data/UCF-Crime-DVS-Test/Fighting/Fighting017_x264.npz"        
    output_folder = "./cropped_frames/Fighting017_x264"
    active_json_path = "temporal_selections/Fighting_active_windows.json"  # From Script 1


    process_all_frames_resize_back(
        frames_folder=frames_folder,
        npz_path=npz_path,
        output_folder=output_folder,
        time_window_us=60000,        
        padding=60,                  
        min_crop_size=(120, 120),    
        target_size="original",      
        save_crop_info=True,         
        active_json_path=active_json_path,
        min_event_density=5.0,       # Start here for anomalies
        use_dense_cluster=True       # Enable for tightest anomaly focus
    )
    
