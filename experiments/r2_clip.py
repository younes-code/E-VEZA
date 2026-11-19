"""
r2_clip.py  --  R8-1b: the event gate in front of a contrastive recogniser (MetaCLIP).

Two steps; step 1 is the only GPU part and is resumable (one .npz per video).

  python r2_clip.py embed --video-root data/UCF-Crime-RGB --eval-list results/eval_videos.txt
  python r2_clip.py eval  --eval-list results/eval_videos.txt

embed : decodes VEZA's 3 s grid (t = 0, 3, 6, ...) from each RGB video, encodes every frame
        with MetaCLIP, stores {t, emb} per video in clip_emb/.  Both arms then come from the
        SAME embeddings, so all-frames vs gated is a paired comparison.
eval  : all frames vs E-VEZA-gated frames (slack rule, published windows, optionally B=500),
        aggregations mean / vote, Top-1/3/5, McNemar, frames kept.

If the student's CLIP benchmark notebook is available, set MODEL / PRETRAINED / PROMPTS
to its values so that the all-frames arm reproduces Table 14 (MetaCLIP vote Top-5 = 80.0 %).
If not, the comparison is still valid because both arms share every setting.
"""
import argparse
import glob
import os
import time

import numpy as np

import r2_common as C

MODEL, PRETRAINED = "ViT-B-32-quickgelu", "metaclip_400m"   # open_clip names
GRID = 3.0
PROMPT_TEMPLATE = "a surveillance camera image of {}"


def class_prompts(n=5):
    """5 descriptions per class from the first terms of Table 4, mean-embedded (as in Sec. 5.7.1)."""
    out = {}
    for c in C.ANOMALY_CLASSES:
        terms = [t.strip() for t in C.DESCRIPTORS[c].split(",")][:n]
        out[c] = [PROMPT_TEMPLATE.format(t) for t in terms]
    return out


def load_model(device):
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    model.eval()
    if device.startswith("cuda"):
        model = model.half()
    tok = open_clip.get_tokenizer(MODEL)
    return model, preprocess, tok, torch


def find_video(root, vid):
    hits = glob.glob(os.path.join(root, "**", vid + ".*"), recursive=True)
    hits = [h for h in hits if h.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]
    return hits[0] if hits else None


def cmd_embed(a):
    import cv2
    from PIL import Image
    model, preprocess, _, torch = load_model(a.device)
    vids = [l.strip() for l in open(a.eval_list) if l.strip()]
    os.makedirs(a.emb_dir, exist_ok=True)
    t_start, n_frames, missing = time.time(), 0, []
    for i, v in enumerate(vids):
        out = os.path.join(a.emb_dir, v + ".npz")
        if os.path.exists(out):
            continue
        path = find_video(a.video_root, v)
        if path is None:
            missing.append(v); continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        times, imgs = [], []
        for t in np.arange(0.0, dur, GRID):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, fr = cap.read()
            if not ok:
                continue
            times.append(float(t))
            imgs.append(preprocess(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
        cap.release()
        if not imgs:
            missing.append(v); continue
        embs = []
        with torch.no_grad():
            for s in range(0, len(imgs), 64):
                x = torch.stack(imgs[s:s + 64]).to(a.device)
                x = x.half() if a.device.startswith("cuda") else x
                e = model.encode_image(x).float()
                embs.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        np.savez(out, t=np.array(times), emb=np.concatenate(embs))
        n_frames += len(times)
        if i % 50 == 0:
            print(f"[{i}/{len(vids)}] {v}: {len(times)} frames  ({time.time()-t_start:.0f}s)")
    print(f"[DONE] {n_frames} frames in {time.time()-t_start:.0f}s; missing videos: {len(missing)}")
    if missing:
        open(os.path.join(a.emb_dir, "_missing.txt"), "w").write("\n".join(missing))


def video_scores(emb, text_emb, agg):
    sims = emb @ text_emb.T                     # frames x classes
    if agg == "mean":
        return sims.mean(0)
    votes = np.bincount(sims.argmax(1), minlength=sims.shape[1]).astype(float)
    return votes + 1e-6 * sims.mean(0)          # vote, ties broken by mean similarity


def cmd_eval(a):
    model, _, tok, torch = load_model(a.device)
    prompts = class_prompts()
    with torch.no_grad():
        T = []
        for c in C.ANOMALY_CLASSES:
            e = model.encode_text(tok(prompts[c]).to(a.device)).float()
            e = e / e.norm(dim=-1, keepdim=True)
            m = e.mean(0)
            T.append((m / m.norm()).cpu().numpy())
    T = np.stack(T)
    vids = [l.strip() for l in open(a.eval_list) if l.strip()]
    vids = [v for v in vids if os.path.exists(os.path.join(a.emb_dir, v + ".npz"))]
    arms = {"gated_B200": C.load_windows(a.windows)}
    if a.npz_root:
        npz = C.find_npz(a.npz_root)
        wc = C.windows_cache(npz, vids, [50, 100, 200, 500], a.windows_cache)   # shared with r2_text bins
        arms["gated_B500"] = {v: ([tuple(w) for w in wc[v]["500"]] if "500" in wc.get(v, {}) else arms["gated_B200"].get(v, []))
                              for v in vids}
    truth = np.array([C.ANOMALY_CLASSES.index(C.video_class(v)) for v in vids])
    res = {"n_videos": len(vids), "model": f"{MODEL}/{PRETRAINED}", "prompts": prompts}
    for agg in ("vote", "mean"):
        ranks, frames = {}, {}
        for arm in ["all"] + list(arms):
            r, nf, empty = [], [], 0
            for v, y in zip(vids, truth):
                d = np.load(os.path.join(a.emb_dir, v + ".npz"))
                t, e = d["t"], d["emb"]
                if arm != "all":
                    keep = np.array([C.keep_slack(x, arms[arm].get(v, [])) for x in t], bool)
                    if keep.sum() == 0:
                        empty += 1
                        keep[:] = True if a.empty_fallback == "all" else False
                    e = e[keep]
                nf.append(len(e))
                if len(e) == 0:
                    r.append(99); continue
                s = video_scores(e, T, agg)
                r.append(int(np.where(np.argsort(-s) == y)[0][0]) + 1)
            ranks[arm], frames[arm] = np.array(r), float(np.mean(nf))
            res[f"{agg}/{arm}"] = {"top1_3_5": C.rak(ranks[arm], Ks=(1, 3, 5)),
                                   "frames_per_video": round(frames[arm], 2),
                                   "videos_with_no_gated_frame": empty}
        for arm in arms:
            res[f"{agg}/{arm}"]["frames_pct_of_all"] = round(100 * frames[arm] / frames["all"], 1)
            for K in (1, 5):
                b, c, p = C.mcnemar_exact(ranks["all"] <= K, ranks[arm] <= K)
                res[f"{agg}/{arm}"][f"mcnemar_top{K}_vs_all"] = {"all_only": b, "gated_only": c, "p": p}
                k = int(np.sum(ranks[arm] <= K))
                res[f"{agg}/{arm}"][f"top{K}_wilson"] = C.wilson(k, len(vids))
        for arm in ["all"] + list(arms):
            print(f"{agg:4s} {arm:11s} Top-1/3/5 {res[f'{agg}/{arm}']['top1_3_5']}  "
                  f"frames/video {res[f'{agg}/{arm}']['frames_per_video']}")
    C.save_json(res, os.path.join(a.out, "clip_gate.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "eval"])
    ap.add_argument("--video-root", default="data/UCF-Crime-RGB")
    ap.add_argument("--eval-list", default="results/eval_videos.txt")
    ap.add_argument("--windows", default="outputs/windows/global_active_windows.json")
    ap.add_argument("--npz-root", default="data/UCF-Crime-DVS", help="'' to skip the B=500 arm")
    ap.add_argument("--emb-dir", default="clip_emb")
    ap.add_argument("--windows-cache", default="results/windows_by_B.json")
    ap.add_argument("--empty-fallback", choices=["none", "all"], default="none",
                    help="video whose gate keeps no frame: count as miss (none) or use all frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"embed": cmd_embed, "eval": cmd_eval}[a.cmd](a)
