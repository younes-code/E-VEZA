# E-VEZA: Efficient Neuromorphic Event-Guided Zero-Shot Video Anomaly Detection

Code, selected windows, generated captions and measured results for the paper

> Y. Kebour, S. Niar, N. Ihaddadene, A. Bekrar, H. Elbez.
> **E-VEZA: Efficient Neuromorphic Event-Guided Zero-Shot Video Anomaly Detection.**
> *Multimedia Tools and Applications* (under review, MTAP-D-26-02821).

E-VEZA puts a training-free event gate in front of a zero-shot video anomaly detector built on
generative vision-language models (VEZA, CBMI 2025). A Dynamic Vision Sensor (DVS) stream is
summarised as a histogram of event counts. Bins above a robust threshold (median + 2·MAD) are
merged into active windows. Only the frames of VEZA's 3 s sampling grid that fall in those
windows are captioned. The captions are then classified zero-shot by sentence-embedding
similarity to 14 class descriptors.

On 938 UCF-Crime-DVS videos (13 anomaly classes), the gate keeps **28.3 %** of VEZA's frames
(a **71.7 %** reduction in VLM calls) at unchanged retrieval accuracy:

| Captioner | VEZA R@10 / R@5 / R@1 | E-VEZA R@10 / R@5 / R@1 | Frames kept |
|---|---|---|---|
| mPLUG-Owl2 | 88.4 / 63.4 / 25.5 | 88.3 / 64.4 / 26.2 | 28.3 % |
| BLIP-2 | 90.7 / 62.9 / 24.0 | 90.5 / 63.9 / 28.8 | 28.0 % |
| GIT | 78.8 / 48.2 / 19.2 | 79.5 / 49.9 / 17.5 | 28.3 % |

```
DVS stream ──► event histogram (B = 200) ──► median + 2·MAD ──► active windows
                                                                     │
RGB video ──► 3 s grid frames ──► keep frames in windows (1 s onset slack) ──► VLM captions
                                                                     │
          NLTK normalisation ──► all-mpnet-base-v2 ──► cosine vs 14 descriptors ──► R@K
```

## Repository layout

| Folder | Contents |
|---|---|
| `gate/` | The event gate: active-window detection from DVS streams, and caption filtering. |
| `captioning/` | Caption generation scripts for mPLUG-Owl2, BLIP-2 and GIT (see `captioning/README.md`). |
| `experiments/` | Analyses reported in the paper: R@K, B sweep, BLIP-2 controls, gate + MetaCLIP, timing, threshold ablation. |
| `evaluation/` | Pre-filter recall, threshold variants, per-module timing, GPU profile. |
| `figures/` | Script that draws the paper figures. |
| `outputs/windows/` | The published active windows (B = 200, median + 2·MAD) for every event stream. |
| `outputs/captions/full/` | VEZA captions: every 3 s grid frame, one file per backbone. |
| `outputs/captions/eveza/` | E-VEZA captions: the subset that falls in the active windows. |
| `results/` | Every measured number, as JSON and logs, for the 938-video set (`results/938/`) and the earlier 888-video set (`results/888/`). |

`results/SUMMARY_tables.md` is generated from the JSON files by `experiments/make_summary.py`,
so it contains no hand-typed numbers.

## Installation

```bash
conda create -n eveza python=3.10 -y && conda activate eveza
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ('punkt','punkt_tab','stopwords','wordnet')]"
```

The versions in `requirements.txt` are those used for the reported results (full freeze:
`results/pip_freeze.txt`). The measurements were made on an NVIDIA RTX 3080 (10 GB) with an
Intel i7-6700K and 16 GB RAM. mPLUG-Owl2 additionally requires its official repository
(see `captioning/README.md`).

## Data

The datasets are not redistributed here. Download them from their authors and place them as follows:

```
data/
├── UCF-Crime-DVS/<Class>/<video>.npz      # event streams (key "t", timestamps in µs)
├── UCF-Crime-RGB/<Class>/<video>.mp4      # original UCF-Crime videos (anomalous classes)
└── Temporal_Anomaly_Annotation.txt        # UCF-Crime test-split temporal annotations
```

* **UCF-Crime-DVS**: Qian et al., AAAI 2025. In the released archive the Assault folder is
  spelled `Assualt`; the evaluation scripts look for both spellings.
* **UCF-Crime**: Sultani et al., CVPR 2018. The temporal annotation file comes with the
  dataset's official test split.

You only need the datasets to recompute windows, timings or CLIP embeddings. The R@K
results can be reproduced from the windows and captions already in `outputs/`.

## Reproducing the paper

Run every command from the repository root.

**1. Active windows** (writes `outputs/windows/`):

```bash
python gate/run_temporal_selection.py
```

**2. E-VEZA captions.** `gate/filter_active_captions.py` keeps a caption at time *t* when
[*t*, *t* + 1 s] overlaps an active window. Set `CAPTIONS_IN` / `CAPTIONS_OUT` at the top of the
script for each backbone, then run:

```bash
python gate/filter_active_captions.py
```

**3. Results, by table and section:**

| Paper | Command | Output |
|---|---|---|
| Table 5 (VEZA vs E-VEZA, R@K) | `python experiments/r2_text.py check --eval-list results/eval938.txt --out results/938` | `results/938/check_table5.json` |
| §5.2 random length-matched control; §5.9 BLIP-2 R@1 (McNemar, no-truncation and random controls) | `python experiments/r2_text.py blip --eval-list results/eval938.txt --out results/938` | `results/938/blip_controls.json` |
| Tables 6–7 (frames, VLM calls, wall-clock) | frame counts from `check_table5.json`; per-frame latency from `python evaluation/gpu_profile.py --image <frame.jpg>` | |
| Table 8 (per-module cost) | `python evaluation/module_timing.py --video data/UCF-Crime-RGB/<Class>/<video>.mp4` | |
| §5.3.2 break-even, MetaCLIP latency and memory (Table 10) | `python experiments/r2_timing.py --video-root data/UCF-Crime-RGB --npz-root data/UCF-Crime-DVS` | `results/timing.json` |
| Table 9 (event gate + MetaCLIP) | `python experiments/r2_clip.py embed --video-root data/UCF-Crime-RGB --eval-list results/eval938.txt`, then `python experiments/r2_clip.py eval --eval-list results/eval938.txt --out results/938` | `results/938/clip_gate.json` |
| Table 11 (pre-filter recall) | `python evaluation/prefilter_recall_gt.py --gt data/Temporal_Anomaly_Annotation.txt` | |
| Table 13 (threshold ablation) | `python experiments/r2_ablation.py --eval-list results/eval938.txt --out results/938` (VLM calls, R@K); `python evaluation/variant_recall.py --gt data/Temporal_Anomaly_Annotation.txt --npz data/UCF-Crime-DVS` (recall, coverage) | `results/938/ablation.json` |
| Table 17 (histogram resolution B) | `python experiments/r2_text.py bins --npz-root data/UCF-Crime-DVS --eval-list results/eval938.txt --out results/938` (R@K, frames); `python evaluation/bin_recall.py --gt data/Temporal_Anomaly_Annotation.txt --npz data/UCF-Crime-DVS` (recall, coverage) | `results/938/bins_endtoend.json` |
| Figures | `python figures/paper_figures.py --json outputs/windows/global_active_windows.json --npz data/UCF-Crime-DVS` | |
| Summary of all tables | `python experiments/make_summary.py` | `results/SUMMARY_tables.md` |

`experiments/make_eval_lists.py` rebuilds the evaluation lists. `results/eval938.txt` holds all
13 classes. `results/eval888.txt` is the earlier set without Assault; the first version of the
manuscript reported results on it, and `r2_text.py check` on it reproduces those numbers exactly.

Run `r2_text.py check` first: the other text-stage experiments share its settings
(`experiments/r2_common.py`, `CONFIG`). Comments inside some scripts refer to table numbers of
earlier manuscript versions. The table above uses the published numbering.

**Not included:** the CLIP-family benchmark of Section 5.7 (Tables 14–16) was run in a
separate code base on the RGB videos and is not part of this repository.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`) and VEZA:

```bibtex
@inproceedings{kebour2025veza,
  title     = {Zero-Shot Vision-Language Model for Event Detection in Smart Surveillance},
  author    = {Kebour, Younes and others},
  booktitle = {2025 International Conference on Content-Based Multimedia Indexing (CBMI)},
  year      = {2025}
}
```

## License

Code: MIT (see `LICENSE`). The generated captions and windows in `outputs/` are released under
the same terms. The UCF-Crime and UCF-Crime-DVS datasets remain under their authors' licenses.
