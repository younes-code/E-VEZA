import cv2
import os
import json

# ----------------------------
# Paths
# ----------------------------
video_path = "data/UCF-Crime-RGB/Shoplifting"
windows_json = "temporal_selections/factive.json"   
output_folder = "data/frames/Shoplifting"

os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# Load activity windows
# ----------------------------
with open(windows_json, "r") as f:
    data = json.load(f)

# your JSON contains a list, get the first entry
activity_windows = data[0]["active_windows"]

# Convert list of [start, end] to Python tuples (in seconds)
activity_windows = [(float(s), float(e)) for s, e in activity_windows]

print("[INFO] Loaded activity windows:")
for w in activity_windows:
    print("   →", w)

# ----------------------------
# Frame extraction with filtering
# ----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_id = 0
saved = 0

def in_active_windows(ts_sec):
    """Return True if ts_sec lies in any (start, end) window."""
    for start, end in activity_windows:
        if start <= ts_sec <= end:
            return True
    return False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # compute timestamp in seconds
    ts_sec = frame_id / fps

    # keep only active frames
    if in_active_windows(ts_sec):
        ts_us = int(ts_sec * 1_000_000)
        save_path = os.path.join(output_folder, f"{ts_us}.jpg")
        cv2.imwrite(save_path, frame)
        saved += 1

    frame_id += 1

cap.release()

print(f"[DONE] Extraction terminée. {saved} frames sauvegardés dans {output_folder}.")
