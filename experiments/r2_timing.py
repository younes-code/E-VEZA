"""
r2_timing.py  --  measurements for the break-even analysis (R8-1a) and the footprint table.

  python r2_timing.py --video-root data/UCF-Crime-RGB --npz-root data/UCF-Crime-DVS

Measures on this machine (close other GPU programs first):
  L_clip   MetaCLIP image-encoder latency per frame, batch 1 and batch 32, fp16, incl. preprocessing
  d        cost of seeking + decoding one grid frame (the per-frame decode term)
  c_g      histogram + median/MAD gate per video (re-check of the 0.9 s)
  peak GPU memory of MetaCLIP
"""
import argparse
import glob
import os
import random
import time

import numpy as np

import r2_common as C
from r2_clip import MODEL, PRETRAINED, load_model


def main(a):
    import cv2
    from PIL import Image
    out = {"machine_note": a.note}
    random.seed(0)
    videos = sorted(glob.glob(os.path.join(a.video_root, "**", "*.mp4"), recursive=True))
    sample = random.sample(videos, min(a.n_videos, len(videos)))

    # d: seek + decode one frame on the 3 s grid
    dec, frames = [], []
    for p in sample:
        cap = cv2.VideoCapture(p)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for t in np.arange(0, n / fps, 3.0)[:20]:
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, fr = cap.read()
            dec.append(time.perf_counter() - t0)
            if ok and len(frames) < 256:
                frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        cap.release()
    out["decode_s_per_frame"] = {"mean": float(np.mean(dec)), "median": float(np.median(dec)), "n": len(dec)}
    print("decode per frame (s):", out["decode_s_per_frame"])

    # L_clip
    model, preprocess, _, torch = load_model(a.device)
    torch.cuda.reset_peak_memory_stats()

    def run(batch, iters):
        ts = []
        for i in range(iters):
            imgs = [frames[(i * batch + j) % len(frames)] for j in range(batch)]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x = torch.stack([preprocess(im) for im in imgs]).to(a.device).half()
            with torch.no_grad():
                model.encode_image(x)
            torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / batch)
        return ts

    run(1, 20); run(32, 3)                                        # warm-up
    b1, b32 = run(1, 200), run(32, 30)
    out["clip"] = {"model": f"{MODEL}/{PRETRAINED}", "precision": "fp16",
                   "s_per_frame_batch1": float(np.median(b1)), "s_per_frame_batch32": float(np.median(b32)),
                   "peak_gpu_MiB": torch.cuda.max_memory_allocated() / 2**20}
    print("MetaCLIP:", out["clip"])

    # c_g: gate time per video, re-check
    npz = list(C.find_npz(a.npz_root).values())
    random.shuffle(npz)
    g = []
    for p in npz[:a.n_videos]:
        t0 = time.perf_counter()
        C.windows_from_npz(p, B=200)
        g.append(time.perf_counter() - t0)
    out["gate_s_per_video"] = {"mean": float(np.mean(g)), "median": float(np.median(g)), "n": len(g)}
    print("gate per video (s):", out["gate_s_per_video"])

    # break-even with the paper's own numbers
    N, r = 37.7, 0.284
    cg, d = out["gate_s_per_video"]["mean"], out["decode_s_per_frame"]["mean"]
    out["break_even_L_ms"] = round(1000 * (cg / ((1 - r) * N) - d), 1)
    print(f"break-even recogniser latency L* = {out['break_even_L_ms']} ms/frame "
          f"(c_g={cg:.3f}s, d={1000*d:.1f}ms, N={N}, r={r})")
    C.save_json(out, os.path.join(a.out, "timing.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", default="data/UCF-Crime-RGB")
    ap.add_argument("--npz-root", default="data/UCF-Crime-DVS")
    ap.add_argument("--n-videos", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--note", default="RTX 3080 10GB, i7-6700K")
    ap.add_argument("--out", default="results")
    main(ap.parse_args())
