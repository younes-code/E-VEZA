#!/usr/bin/env python3
"""git_provenance.py -- which GIT checkpoint produced outputs/captions/full/git_captions.txt?
Re-captions N frames that appear in the caption file with microsoft/git-base-coco and
microsoft/git-large-coco (greedy, max_length=50, as in ~/Projects/GIT/caption_generation.py)
and counts exact matches against the stored captions. Writes results/git_provenance.json."""
import json, os, random, re, sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FR = os.path.join(ROOT, "data/UCF_Crimes_Frames")
LINE = re.compile(r"^(\S+)/(\d+) (\d+:\d+:\d+)\s+##\s?(.*)$")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12

rows = [m.groups() for m in map(LINE.match, open(os.path.join(ROOT, "outputs/captions/full/git_captions.txt"), errors="ignore")) if m]
random.seed(0)
random.shuffle(rows)
items = []
for vid, idx, ts, cap in rows:
    cls = re.match(r"[A-Za-z]+", vid).group(0)
    p = os.path.join(FR, cls, vid, f"{idx} {ts}.jpg")
    if os.path.exists(p):
        items.append((vid, idx, ts, cap.strip(), p))
    if len(items) == N:
        break
print(f"{len(items)} frames found")
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = {"n_frames": len(items), "frames": [i[4] for i in items], "models": {}}
for name in ("microsoft/git-base-coco", "microsoft/git-large-coco"):
    proc = AutoProcessor.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(dev).eval()
    exact, gen = 0, []
    for vid, idx, ts, cap, p in items:
        x = proc(images=Image.open(p).convert("RGB"), return_tensors="pt").to(dev)
        with torch.no_grad():
            ids = model.generate(**x, max_length=50)
        g = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        exact += (g.lower() == cap.lower())
        gen.append({"stored": cap, "generated": g})
        print(f"[{name.split('/')[1]}] {vid}/{idx}\n   stored: {cap}\n   gen   : {g}")
    out["models"][name] = {"exact_matches": exact, "pairs": gen}
    print(f"== {name}: {exact}/{len(items)} exact matches")
    del model
    torch.cuda.empty_cache()
json.dump(out, open(os.path.join(ROOT, "results/git_provenance.json"), "w"), indent=2)
print("[SAVED] results/git_provenance.json")
