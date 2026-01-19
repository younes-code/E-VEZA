import os
import json
import cv2
from glob import glob

# =============================
# CONFIGURATION
# =============================
JSON_ROOT = "temporal_selections"        # Folder where the .json exist
RGB_ROOT  = "data/UCF-Crime-RGB"         # Folder with RGB videos
OUTPUT_ROOT = "active_frames_output"     # Where extracted frames will be stored

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =============================
# Helper: Check if timestamp is inside any interval
# =============================
def in_active_windows(timestamp, active_windows):
    for (start, end) in active_windows:
        if start <= timestamp <= end:
            return True
    return False


# =============================
# Process ONE JSON file
# =============================
def process_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    video_name = data["video_name"]                        # Example: Abuse001_x264
    active_windows = data["active_windows"]

    # Identify class = parent folder name
    class_name = os.path.basename(os.path.dirname(json_path))

    # Corresponding RGB video path
    rgb_video_path = os.path.join(RGB_ROOT, class_name, video_name + ".mp4")

    if not os.path.exists(rgb_video_path):
        print(f"[WARNING] RGB video not found: {rgb_video_path}")
        return

    print(f"[INFO] Processing: {rgb_video_path}")

    # Output dir
    out_dir = os.path.join(OUTPUT_ROOT, class_name, video_name)
    os.makedirs(out_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(rgb_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Timestamp of current frame
        timestamp = frame_idx / fps

        if in_active_windows(timestamp, active_windows):
            frame_path = os.path.join(out_dir, f"{video_name}_frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f" → Saved {saved} frames for {video_name}")


# =============================
# MAIN LOOP
# =============================
if __name__ == "__main__":
    # Find all per-video JSONs inside class folders.
    json_files = glob(os.path.join(JSON_ROOT, "*", "*_windows.json"))

    print(f"[INFO] Found {len(json_files)} JSON files")

    for json_path in json_files:
        process_json(json_path)

    print("\n[DONE] All frames extracted.")
