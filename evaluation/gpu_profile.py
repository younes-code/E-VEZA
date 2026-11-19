#!/usr/bin/env python3
"""
gpu_profile.py — the R5-#4 columns that are actually measurable.
Run on the RTX 3080 machine.

    python evaluation/gpu_profile.py --image path/to/any/frame.jpg

Reports per captioning backbone:
    parameters          sum(p.numel())
    peak GPU memory     torch.cuda.max_memory_allocated() around one caption
    latency             seconds for that one caption (sanity check vs Table 7)

Plus the sentence encoder, which is the only model E-VEZA adds on top.

Loads one model at a time and frees it, so 10 GB is enough even for mPLUG-2.
If a model OOMs or the checkpoint is missing, it says so and continues —
a partial table is fine, a wrong one is not.

FOR GPU/CPU UTILISATION, run this in a second terminal during your normal
captioning job and take the median of the printed column:

    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used \
               --format=csv -l 1 > gpu_util.csv
"""
import argparse, gc, sys, time

MODELS = [
    ("mPLUG-2", "MAGAer13/mplug-owl2-llama2-7b", "fp16"),
    ("BLIP-2",  "Salesforce/blip2-opt-2.7b",     "4bit"),
    ("GIT",     "microsoft/git-base-coco",       "4bit"),
]
ENCODER = "sentence-transformers/all-mpnet-base-v2"


def human(n):
    for u in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{u}"
        n /= 1000.0
    return f"{n:.1f}T"


def free():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def try_caption(name, ckpt, mode, image):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq

    free()
    t0 = time.perf_counter()
    try:
        kw = {}
        if mode == "fp16":
            kw["torch_dtype"] = torch.float16
        elif mode == "4bit":
            kw["load_in_4bit"] = True
        proc = AutoProcessor.from_pretrained(ckpt)
        model = AutoModelForVision2Seq.from_pretrained(ckpt, device_map="auto",
                                                       **kw)
    except Exception as e:
        print(f"{name:<10} could not load: {type(e).__name__}: {e}")
        return None
    load_s = time.perf_counter() - t0

    params = sum(p.numel() for p in model.parameters())
    img = Image.open(image).convert("RGB")
    w, h = img.size
    if max(w, h) > 1024:
        s = 1024.0 / max(w, h)
        img = img.resize((int(w * s), int(h * s)))

    try:
        inputs = proc(images=img, return_tensors="pt").to(model.device)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=50)
        torch.cuda.synchronize()
        lat = time.perf_counter() - t1
        peak = torch.cuda.max_memory_allocated() / 2**30
    except Exception as e:
        print(f"{name:<10} loaded ({human(params)} params) but generate "
              f"failed: {type(e).__name__}")
        del model; free()
        return (name, params, None, None, load_s)

    del model, proc
    free()
    return (name, params, peak, lat, load_s)


def encoder_row():
    import torch
    from sentence_transformers import SentenceTransformer
    free()
    m = SentenceTransformer(ENCODER)
    params = sum(p.numel() for p in m.parameters())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        m = m.to("cuda")
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    m.encode(["a group of people in a parking lot "] * 17)
    if dev == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
    else:
        peak = None
    lat = time.perf_counter() - t0
    del m; free()
    return ("Sentence encoder", params, peak, lat, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="any extracted frame")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="model names to skip, e.g. --skip mPLUG-2")
    a = ap.parse_args()

    try:
        import torch
        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
              f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}\n")
    except Exception:
        sys.exit("torch not importable")

    rows = []
    for name, ckpt, mode in MODELS:
        if name in a.skip:
            continue
        print(f"[{name}] loading {ckpt} ({mode}) ...")
        r = try_caption(name, ckpt, mode, a.image)
        if r:
            rows.append(r)
    try:
        rows.append(encoder_row())
    except Exception as e:
        print(f"encoder failed: {e}")

    print("\n" + "=" * 70)
    print("R5-#4 — measurable columns")
    print("=" * 70)
    print(f"{'Model':<18}{'Params':>10}{'Peak GPU (GiB)':>17}{'Latency (s)':>14}")
    print("-" * 70)
    for name, params, peak, lat, _ in rows:
        p = f"{peak:.2f}" if peak is not None else "n/a"
        l = f"{lat:.2f}" if lat is not None else "n/a"
        print(f"{name:<18}{human(params):>10}{p:>17}{l:>14}")

    print("\nNOTE: latency here is ONE caption on ONE frame and will not match"
          "\nTable 7 exactly (different max_new_tokens, no batching). Use it as"
          "\na sanity check, NOT as a replacement for Table 7's figures.")
    print("\nSend me this whole block.")


if __name__ == "__main__":
    main()
