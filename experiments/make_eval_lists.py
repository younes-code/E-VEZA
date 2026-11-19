#!/usr/bin/env python3
"""
make_eval_lists.py -- rebuild the evaluation lists exactly as scripts_rev/variant_rak.py does.

eval888.txt : variant_rak logic verbatim (find_npz never tries the misspelled 'Assualt' folder).
eval938.txt : identical logic, except find_npz also tries 'Assualt' (so Assault videos are found).

Selection (copied from variant_rak.py main() + load_counts()):
  1. video in global_active_windows.json AND in the mPLUG caption file (LINE regex)
  2. class prefix from ^([A-Za-z]+?)\\d+_x264, mapped RoadAccidents->Accident, must be a descriptor
  3. npz found by find_npz, zipfile.is_zipfile, np.load(p)["t"] loads, t.size>0 and t.max()>t.min()
  sorted alphabetically.
Step 3 is evaluated with the cached result (variant_counts_cache.npz, written by variant_rak.py
itself with this exact test) for videos in the cache, and by actually loading "t" for the others.
"""
import json, os, re, sys, zipfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
LINE = re.compile(r"^(.+?)/\d+\s+(\d{1,3}:\d{2}:\d{3})(?:\.jpg)?\s+##\s?(.*)$")
NAME_CLASS = re.compile(r"^([A-Za-z]+?)\d+_x264")
CAND = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "Accident",
        "Normal", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"]
CLASS_TO_CAND = {"RoadAccidents": "Accident"}
NPZ = "data/UCF-Crime-DVS"


def find_npz(cls, name, assualt):
    tries = [cls, cls.replace("RoadAccidents", "Road_Accidents"),
             cls.replace("RoadAccidents", "RoadAccident"), cls.lower()]
    if assualt:
        tries.append(cls.replace("Assault", "Assualt"))
    for c in tries:
        p = os.path.join(NPZ, c, name + ".npz")
        if os.path.exists(p) and zipfile.is_zipfile(p):
            return p
    return None


caps = set()
with open("outputs/captions/full/mplug_captions.txt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = LINE.match(line.rstrip("\n"))
        if m:
            caps.add(m.group(1))
pub = {r["video_name"] for r in json.load(open("outputs/windows/global_active_windows.json"))}
cands = []
for v in sorted(pub):
    if v not in caps:
        continue
    m = NAME_CLASS.match(v)
    if not m:
        continue
    cls = m.group(1)
    if CLASS_TO_CAND.get(cls, cls) not in CAND:
        continue
    cands.append((v, cls))
print(f"JSON & captions & class ok: {len(cands)}")

cache = set(np.load("variant_counts_cache.npz", allow_pickle=True)["names"].tolist())
for tag, assualt in (("888", False), ("938", True)):
    keep, loaded = [], 0
    for v, cls in cands:
        p = find_npz(cls, v, assualt)
        if p is None:
            continue
        if v in cache:
            keep.append(v)
            continue
        loaded += 1
        try:
            t = np.load(p)["t"] / 1e6
        except Exception as e:
            print(f"  [{tag}] unreadable t: {v}: {e}")
            continue
        if t.size == 0 or t.max() <= t.min():
            print(f"  [{tag}] empty/flat stream: {v}")
            continue
        keep.append(v)
    keep.sort()
    with open(f"results/eval{tag}.txt", "w") as f:
        f.write("\n".join(keep) + "\n")
    print(f"eval{tag}.txt: {len(keep)} videos ({loaded} streams loaded outside the cache)")
    if tag == "888":
        assert set(keep) == cache, "888 list differs from variant_counts_cache.npz names"
        print("  identical to the 888 names in variant_counts_cache.npz")
