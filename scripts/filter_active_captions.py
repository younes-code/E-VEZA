import json
import re

# =========================
# CONFIG
# =========================
GLOBAL_JSON = "temporal_selections/global_temporal_selection.json"
CAPTIONS_IN = "data/git_captions.txt"
CAPTIONS_OUT = "temporal_selections/git_active_captions.txt"


# =========================
# Helpers
# =========================
def parse_timestamp(ts_str):
    """
    Convert MM:SS:ms → seconds
    Example: 00:06:000 → 6.0
    """
    try:
        m, s, ms = ts_str.split(":")
        return int(m) * 60 + int(s) + int(ms) / 1000.0
    except Exception:
        return None


def is_in_active_windows(t, windows):
    caption_start = t
    caption_end = t + 1.0  # whole second

    for start, end in windows:
        if caption_end >= start and caption_start <= end:
            return True
    return False



# =========================
# Load global temporal selections
# =========================
with open(GLOBAL_JSON, "r") as f:
    global_data = json.load(f)

# Build lookup: video_name → active_windows
video_to_windows = {}
for item in global_data:
    video = item.get("video_name")
    windows = item.get("active_windows")
    if video and windows:
        video_to_windows[video] = windows

print(f"[INFO] Loaded active windows for {len(video_to_windows)} videos")


# =========================
# Process captions
# =========================
kept = 0
total = 0
skipped_no_video = 0
skipped_no_overlap = 0

# # STRICT regex (fully anchored)
# caption_pattern = re.compile(
#     r"^(.+?)/\d+\s+(\d{2}:\d{2}:\d{3})\.jpg\s+##"
# )

# STRICT regex for git only ( without jpg )
caption_pattern = re.compile(
    r"^(.+?)/\d+\s+(\d{2}:\d{2}:\d{3})\s+##"
)

with open(CAPTIONS_IN, "r", encoding="utf-8") as fin, \
     open(CAPTIONS_OUT, "w", encoding="utf-8") as fout:

    for line in fin:
        total += 1

        match = caption_pattern.match(line)
        if not match:
            continue

        video_name = match.group(1)
        timestamp = parse_timestamp(match.group(2))

        if timestamp is None:
            continue

        # ---- STRICT video binding ----
        windows = video_to_windows.get(video_name)
        if windows is None:
            skipped_no_video += 1
            continue

        if not is_in_active_windows(timestamp, windows):
            skipped_no_overlap += 1
            continue

        fout.write(line)
        kept += 1

print("\n[DONE]")
print(f"  Total captions processed : {total}")
print(f"  Captions kept            : {kept}")
print(f"  Skipped (no video)       : {skipped_no_video}")
print(f"  Skipped (no overlap)     : {skipped_no_overlap}")
print(f"[SAVED] {CAPTIONS_OUT}")
