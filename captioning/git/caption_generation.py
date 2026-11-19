# Caption every extracted frame with GIT.
# The released captions (outputs/captions/full/git_captions.txt) were produced with
# microsoft/git-base-coco; see experiments/git_provenance.py for the check.
import argparse
import gc
import os
import re

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def extract_frame_number(filename):
    """Sort key: the first integer in the file name."""
    m = re.search(r"\d+", filename)
    return int(m.group()) if m else -1


def generate_caption(model, processor, img_path, device):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_length=50)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main(frames_dir, save_path, model_name="microsoft/git-base-coco"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    for root, _, files in os.walk(frames_dir):
        for name in sorted((f for f in files if f.lower().endswith(IMAGE_EXT)), key=extract_frame_number):
            caption = generate_caption(model, processor, os.path.join(root, name), device)
            # line format: <video>/<frame id> ## <caption>
            frame_id = os.path.splitext(name)[0].split("_", 1)[-1]
            with open(save_path, "a", encoding="utf-8") as f:
                f.write(f"{os.path.basename(root)}/{frame_id} ## {caption}\n")

    del model, processor
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Caption extracted frames with GIT.")
    ap.add_argument("--frames-dir", required=True, help="folder of extracted frames (one sub-folder per video)")
    ap.add_argument("--out", default="git_captions.txt", help="caption file to append to")
    ap.add_argument("--model", default="microsoft/git-base-coco")
    args = ap.parse_args()
    main(args.frames_dir, args.out, args.model)
