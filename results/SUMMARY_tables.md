## 1. Table 5 calibration (`check_table5.json`)

R@10 / R@5 / R@1. 888 = Table 5 set (calibration target, must match); 938 = same logic + Assault.

| Backbone | Set | VEZA | E-VEZA | Table 5 VEZA | Table 5 E-VEZA | frames kept % | E-VEZA videos w/o caption |
|---|---|---|---|---|---|---|---|
| mPLUG(-Owl2) | 888 (N=888) | 88.7 / 65.4 / 26.8 | 88.9 / 66.0 / 27.3 | 88.7 / 65.4 / 26.8 | 88.9 / 66.0 / 27.3 | 28.4 | 1 |
| BLIP-2 | 888 (N=888) | 91.7 / 65.5 / 25.1 | 92.5 / 65.7 / 30.0 | 91.7 / 65.5 / 25.1 | 92.5 / 65.7 / 30.0 | 28.1 | 0 |
| GIT | 888 (N=888) | 80.3 / 50.3 / 20.3 | 81.1 / 52.0 / 18.5 | 80.3 / 50.3 / 20.3 | 81.1 / 52.0 / 18.5 | 28.4 | 1 |
| mPLUG(-Owl2) | 938 (N=938) | 88.4 / 63.4 / 25.5 | 88.3 / 64.4 / 26.2 | 88.7 / 65.4 / 26.8 | 88.9 / 66.0 / 27.3 | 28.3 | 2 |
| BLIP-2 | 938 (N=938) | 90.7 / 62.9 / 24.0 | 90.5 / 63.9 / 28.8 | 91.7 / 65.5 / 25.1 | 92.5 / 65.7 / 30.0 | 28.0 | 0 |
| GIT | 938 (N=938) | 78.8 / 48.2 / 19.2 | 79.5 / 49.9 / 17.5 | 80.3 / 50.3 / 20.3 | 81.1 / 52.0 / 18.5 | 28.3 | 2 |

## 2. Timing (`timing.json`)

| Quantity | Value |
|---|---|
| Machine | RTX 3080 10GB, i7-6700K 4.00GHz, 16 GB RAM |
| MetaCLIP model | ViT-B-32-quickgelu/metaclip_400m (fp16) |
| MetaCLIP image encoder, batch 1 (incl. preprocessing) | 7.50 ms/frame |
| MetaCLIP image encoder, batch 32 (incl. preprocessing) | 2.04 ms/frame |
| MetaCLIP peak GPU memory (torch allocated) | 357.9 MiB |
| Seek + decode one 3 s-grid frame (OpenCV) | mean 16.49 ms, median 15.52 ms (n=871) |
| Gate per video (load npz + B=200 histogram + median/MAD) | mean 1.061 s, median 0.709 s (n=50) |
| Break-even recogniser latency L* = c_g/((1-r)N) - d, N=37.7, r=0.284 | 22.8 ms/frame |

## 3. B sweep end to end, mPLUG captions (`bins_endtoend.json`)

**888 videos (N=888)** — VEZA (all frames): R@10/5/1 88.7 / 65.4 / 26.8, 37.74 captions/video. Recomputed B=200 windows identical to the published ones for 888/888 videos.

| B | R@10 / R@5 / R@1 | captions/video | % of VEZA frames | windows/video | no-caption videos | R@10 95% Wilson | McNemar vs B200 R@1 (B200-only / this-only / p) | R@5 | R@10 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 87.4 / 63.5 / 29.2 | 8.67 | 23.0 | 4.25 | 11 | 85.0–89.4 | 28/45/0.0604 | 47/25/0.0128 | 29/16/0.0725 |
| 100 | 88.0 / 65.0 / 28.3 | 9.7 | 25.7 | 6.58 | 3 | 85.6–89.9 | 17/26/0.222 | 28/19/0.243 | 17/9/0.169 |
| 200 | 88.9 / 66.0 / 27.3 | 10.7 | 28.4 | 10.2 | 1 | 86.6–90.8 | 0/0/1 | 0/0/1 | 0/0/1 |
| 500 | 89.3 / 66.8 / 26.7 | 12.35 | 32.7 | 19.9 | 1 | 87.1–91.2 | 21/16/0.511 | 17/24/0.349 | 11/15/0.557 |

**938 videos (N=938)** — VEZA (all frames): R@10/5/1 88.4 / 63.4 / 25.5, 37.3 captions/video. Recomputed B=200 windows identical to the published ones for 938/938 videos.

| B | R@10 / R@5 / R@1 | captions/video | % of VEZA frames | windows/video | no-caption videos | R@10 95% Wilson | McNemar vs B200 R@1 (B200-only / this-only / p) | R@5 | R@10 |
|---|---|---|---|---|---|---|---|---|---|
| 50 | 86.6 / 61.5 / 27.8 | 8.51 | 22.8 | 4.22 | 14 | 84.2–88.6 | 30/45/0.105 | 53/26/0.00318 | 35/19/0.0402 |
| 100 | 87.5 / 63.1 / 27.1 | 9.52 | 25.5 | 6.54 | 4 | 85.3–89.5 | 18/26/0.291 | 33/21/0.134 | 18/11/0.265 |
| 200 | 88.3 / 64.4 / 26.2 | 10.54 | 28.3 | 10.16 | 2 | 86.1–90.2 | 0/0/1 | 0/0/1 | 0/0/1 |
| 500 | 88.9 / 65.0 / 25.5 | 12.16 | 32.6 | 19.76 | 1 | 86.7–90.8 | 23/16/0.337 | 19/25/0.451 | 11/17/0.345 |

## 4. Backbone controls (`blip_controls.json`)

**888 videos (N=888)**

| | mPLUG(-Owl2) | BLIP-2 | GIT |
|---|---|---|---|
| VEZA R@10/5/1 | 88.7 / 65.4 / 26.8 | 91.7 / 65.5 / 25.1 | 80.3 / 50.3 / 20.3 |
| E-VEZA R@10/5/1 | 88.9 / 66.0 / 27.3 | 92.5 / 65.7 / 30.0 | 81.1 / 52.0 / 18.5 |
| McNemar R@1 (VEZA-only / E-VEZA-only / p) | 43 / 47 / 0.752 | 44 / 87 / 0.000216 | 51 / 35 / 0.105 |
| McNemar R@5 (VEZA-only / E-VEZA-only / p) | 53 / 58 / 0.704 | 55 / 56 / 1 | 51 / 66 / 0.195 |
| McNemar R@10 (VEZA-only / E-VEZA-only / p) | 30 / 31 / 1 | 17 / 24 / 0.349 | 53 / 60 / 0.573 |
| captions/video VEZA → E-VEZA | 37.7 → 10.7 | 140.6 → 39.5 | 37.7 → 10.7 |
| median tokens VEZA / E-VEZA | 626 / 188 | 508 / 155 | 130 / 43 |
| % docs truncated at 512 VEZA / E-VEZA | 59.6 / 12.6 | 49.8 / 11.7 | 8.6 / 0.2 |
| sampling interval median (s); videos ≤1.5 s / ≥2.5 s | 3.0; 0 / 888 | 1.0; 791 / 97 | 3.0; 0 / 888 |
| random length-matched, 10 seeds, mean (R@10/5/1) | 88.4 / 64.1 / 27.2 | 91.5 / 65.2 / 28.8 | 80.6 / 49.4 / 18.0 |
|   sd | 0.4 / 0.8 / 0.5 | 0.3 / 0.4 / 0.8 | 0.7 / 0.8 / 0.6 |
|   min / max R@1 | 26.1 / 27.8 | 27.6 / 30.2 | 17.3 / 19.1 |
| no-truncation (chunked) VEZA | 89.8 / 65.2 / 27.7 | 91.8 / 66.4 / 26.6 | 80.9 / 50.6 / 20.3 |
| no-truncation (chunked) E-VEZA | 88.6 / 66.0 / 27.4 | 92.5 / 65.3 / 30.6 | 81.1 / 52.0 / 18.5 |
| chunked McNemar R@1 (VEZA-only / E-VEZA-only / p) | 50 / 47 / 0.839 | 42 / 78 / 0.0013 | 49 / 33 / 0.097 |

**938 videos (N=938)**

| | mPLUG(-Owl2) | BLIP-2 | GIT |
|---|---|---|---|
| VEZA R@10/5/1 | 88.4 / 63.4 / 25.5 | 90.7 / 62.9 / 24.0 | 78.8 / 48.2 / 19.2 |
| E-VEZA R@10/5/1 | 88.3 / 64.4 / 26.2 | 90.5 / 63.9 / 28.8 | 79.5 / 49.9 / 17.5 |
| McNemar R@1 (VEZA-only / E-VEZA-only / p) | 43 / 50 / 0.534 | 45 / 90 / 0.000134 | 51 / 35 / 0.105 |
| McNemar R@5 (VEZA-only / E-VEZA-only / p) | 57 / 66 / 0.471 | 57 / 66 / 0.471 | 52 / 68 / 0.171 |
| McNemar R@10 (VEZA-only / E-VEZA-only / p) | 34 / 33 / 1 | 29 / 27 / 0.894 | 61 / 68 / 0.597 |
| captions/video VEZA → E-VEZA | 37.3 → 10.5 | 138.4 → 38.7 | 37.3 → 10.5 |
| median tokens VEZA / E-VEZA | 618 / 184 | 492 / 151 | 130 / 42 |
| % docs truncated at 512 VEZA / E-VEZA | 59.1 / 12.5 | 49.3 / 11.4 | 8.4 / 0.2 |
| sampling interval median (s); videos ≤1.5 s / ≥2.5 s | 3.0; 0 / 938 | 1.0; 832 / 106 | 3.0; 0 / 938 |
| random length-matched, 10 seeds, mean (R@10/5/1) | 87.3 / 62.4 / 25.8 | 89.8 / 63.1 / 27.3 | 79.2 / 47.1 / 17.1 |
|   sd | 0.9 / 0.7 / 0.6 | 0.6 / 0.7 / 0.8 | 0.8 / 1.0 / 0.7 |
|   min / max R@1 | 24.7 / 27.0 | 25.5 / 28.4 | 15.9 / 18.4 |
| no-truncation (chunked) VEZA | 89.6 / 63.8 / 26.3 | 90.7 / 63.9 / 25.4 | 79.3 / 48.4 / 19.2 |
| no-truncation (chunked) E-VEZA | 88.1 / 64.3 / 26.3 | 90.5 / 63.6 / 29.4 | 79.5 / 49.9 / 17.5 |
| chunked McNemar R@1 (VEZA-only / E-VEZA-only / p) | 50 / 50 / 1 | 43 / 81 / 0.00082 | 49 / 33 / 0.097 |

## 5. Event gate + MetaCLIP (`clip_gate.json`)

**888 videos (N=888)**, model ViT-B-32-quickgelu/metaclip_400m, 13 anomaly classes, empty gate = miss

| Aggregation | Arm | Top-1 / 3 / 5 | frames/video | % of all frames | videos with no gated frame | McNemar Top-1 vs all (all-only / gated-only / p) | McNemar Top-5 | Top-1 Wilson | Top-5 Wilson |
|---|---|---|---|---|---|---|---|---|---|
| vote | all | 32.3 / 57.8 / 71.7 | 37.75 | 100 | – | – | – | – | – |
| vote | gated_B200 | 32.1 / 57.5 / 71.4 | 10.7 | 28.4 | 1 | 30/28/0.896 | 24/21/0.766 | 29.1–35.2 | 68.3–74.3 |
| vote | gated_B500 | 32.2 / 57.8 / 71.3 | 12.35 | 32.7 | 1 | 26/25/1 | 22/18/0.636 | 29.2–35.4 | 68.2–74.2 |
| mean | all | 31.6 / 55.9 / 69.9 | 37.75 | 100 | – | – | – | – | – |
| mean | gated_B200 | 32.1 / 56.2 / 70.3 | 10.7 | 28.4 | 1 | 27/31/0.694 | 26/29/0.788 | 29.1–35.2 | 67.2–73.2 |
| mean | gated_B500 | 32.3 / 55.6 / 70.2 | 12.35 | 32.7 | 1 | 20/26/0.461 | 22/24/0.883 | 29.3–35.5 | 67.1–73.1 |

**938 videos (N=938)**, model ViT-B-32-quickgelu/metaclip_400m, 13 anomaly classes, empty gate = miss

| Aggregation | Arm | Top-1 / 3 / 5 | frames/video | % of all frames | videos with no gated frame | McNemar Top-1 vs all (all-only / gated-only / p) | McNemar Top-5 | Top-1 Wilson | Top-5 Wilson |
|---|---|---|---|---|---|---|---|---|---|
| vote | all | 30.8 / 56.0 / 71.3 | 37.3 | 100 | – | – | – | – | – |
| vote | gated_B200 | 30.9 / 56.5 / 71.0 | 10.54 | 28.3 | 2 | 30/31/1 | 28/25/0.784 | 28.0–33.9 | 68.0–73.8 |
| vote | gated_B500 | 30.9 / 56.6 / 70.8 | 12.16 | 32.6 | 1 | 26/27/1 | 25/20/0.551 | 28.0–33.9 | 67.8–73.6 |
| mean | all | 30.2 / 54.7 / 69.9 | 37.3 | 100 | – | – | – | – | – |
| mean | gated_B200 | 30.8 / 55.3 / 70.0 | 10.54 | 28.3 | 2 | 27/33/0.519 | 30/31/1 | 27.9–33.8 | 67.0–72.9 |
| mean | gated_B500 | 30.9 / 54.7 / 70.0 | 12.16 | 32.6 | 1 | 20/27/0.382 | 25/26/1 | 28.0–33.9 | 67.0–72.9 |

Prompts (per class, mean of 5 normalised text embeddings):

- Abuse: a surveillance camera image of abuse; a surveillance camera image of maltreatment; a surveillance camera image of mistreatment; a surveillance camera image of cruelty; a surveillance camera image of harm
- Arrest: a surveillance camera image of arrest; a surveillance camera image of detain; a surveillance camera image of apprehend; a surveillance camera image of capture; a surveillance camera image of take into custody
- Arson: a surveillance camera image of arson; a surveillance camera image of fire-raising; a surveillance camera image of incendiarism; a surveillance camera image of pyromania; a surveillance camera image of torching
- Assault: a surveillance camera image of assault; a surveillance camera image of attack; a surveillance camera image of physical attack; a surveillance camera image of aggression; a surveillance camera image of violence
- Burglary: a surveillance camera image of burglary; a surveillance camera image of break-in; a surveillance camera image of robbery; a surveillance camera image of housebreaking; a surveillance camera image of theft
- Explosion: a surveillance camera image of explosion; a surveillance camera image of blast; a surveillance camera image of detonation; a surveillance camera image of eruption; a surveillance camera image of blowup
- Fighting: a surveillance camera image of fighting; a surveillance camera image of brawling; a surveillance camera image of combat; a surveillance camera image of conflict; a surveillance camera image of struggle
- Accident: a surveillance camera image of accident; a surveillance camera image of crash; a surveillance camera image of collision; a surveillance camera image of traffic accident; a surveillance camera image of car accident
- Robbery: a surveillance camera image of robbery; a surveillance camera image of theft; a surveillance camera image of heist; a surveillance camera image of larceny; a surveillance camera image of hold-up
- Shooting: a surveillance camera image of shooting; a surveillance camera image of gunfire; a surveillance camera image of gunfight; a surveillance camera image of gunplay; a surveillance camera image of firing
- Shoplifting: a surveillance camera image of shoplifting; a surveillance camera image of theft; a surveillance camera image of pilfering; a surveillance camera image of larceny; a surveillance camera image of stealing
- Stealing: a surveillance camera image of stealing; a surveillance camera image of theft; a surveillance camera image of larceny; a surveillance camera image of robbery; a surveillance camera image of pilfering
- Vandalism: a surveillance camera image of vandalism; a surveillance camera image of destruction; a surveillance camera image of defacement; a surveillance camera image of sabotage; a surveillance camera image of graffiti

## 6. Threshold ablation, paper Table 12 (`ablation.json`)

mPLUG captions, slack rule, B=200, windows recomputed from the event-stream histograms; same CONFIG as the check. **edges rel** = `variant_rak.py` convention (bin edges from the first event, reproduces rak_final.txt); **edges abs** = `np.histogram` absolute edges as in `run_temporal_selection.py` (MAD/abs = the published windows). McNemar vs MAD within the same convention (MAD-only / this-only / p).

**888 videos (N=888)**: VEZA 88.7 / 65.4 / 26.8; published windows 88.9 / 66.0 / 27.3. Histograms: 888 from `variant_counts_cache.npz`, 0 loaded from npz; cache re-verified against full stream reloads 20/20. abs-edge MAD windows identical to published: 888/888.

| Edges | Threshold | R@10 / R@5 / R@1 | captions/video | % of VEZA | no-caption videos | windows/video | McNemar R@1 vs MAD | McNemar R@5 vs MAD | McNemar R@10 vs MAD |
|---|---|---|---|---|---|---|---|---|---|
| rel | median + 2·MAD | 88.7 / 66.0 / 27.3 | 10.7 | 28.3 | 1 | 10.2 | 0/0/1 | 0/0/1 | 0/0/1 |
| rel | mean + 1σ | 87.3 / 63.2 / 26.9 | 6.21 | 16.5 | 17 | 7.55 | 44/41/0.828 | 59/34/0.0124 | 40/27/0.142 |
| rel | 80th percentile | 89.2 / 66.2 / 26.7 | 10.74 | 28.4 | 2 | 11.36 | 23/18/0.533 | 21/23/0.88 | 12/16/0.572 |
| rel | median | 89.4 / 66.0 / 27.5 | 22.53 | 59.7 | 0 | 14.4 | 40/42/0.912 | 44/44/1 | 21/27/0.471 |
| abs | median + 2·MAD | 88.9 / 66.0 / 27.3 | 10.7 | 28.4 | 1 | 10.2 | 0/0/1 | 0/0/1 | 0/0/1 |
| abs | mean + 1σ | 87.4 / 63.3 / 27.0 | 6.22 | 16.5 | 16 | 7.55 | 44/42/0.914 | 58/34/0.016 | 39/26/0.136 |
| abs | 80th percentile | 89.2 / 66.2 / 26.7 | 10.74 | 28.5 | 2 | 11.36 | 23/18/0.533 | 21/23/0.88 | 12/15/0.701 |
| abs | median | 89.3 / 66.0 / 27.5 | 22.53 | 59.7 | 0 | 14.4 | 40/42/0.912 | 44/44/1 | 22/26/0.665 |

Reproduction of `rak_final.txt` (slack, edges rel): **all four variants match** (R@10/5/1, no-caption videos and captions/video).

| Threshold | got | rak_final.txt | no-caption got / rak_final | captions/video got / rak_final |
|---|---|---|---|---|
| median + 2·MAD | 88.7 / 66.0 / 27.3 | 88.7 / 66.0 / 27.3 | 1 / 1 | 10.7 / 10.7 |
| mean + 1σ | 87.3 / 63.2 / 26.9 | 87.3 / 63.2 / 26.9 | 17 / 17 | 6.2 / 6.2 |
| 80th percentile | 89.2 / 66.2 / 26.7 | 89.2 / 66.2 / 26.7 | 2 / 2 | 10.7 / 10.7 |
| median | 89.4 / 66.0 / 27.5 | 89.4 / 66.0 / 27.5 | 0 / 0 | 22.5 / 22.5 |

**938 videos (N=938)**: VEZA 88.4 / 63.4 / 25.5; published windows 88.3 / 64.4 / 26.2. Histograms: 888 from `variant_counts_cache.npz`, 50 loaded from npz; cache re-verified against full stream reloads 20/20. abs-edge MAD windows identical to published: 938/938.

| Edges | Threshold | R@10 / R@5 / R@1 | captions/video | % of VEZA | no-caption videos | windows/video | McNemar R@1 vs MAD | McNemar R@5 vs MAD | McNemar R@10 vs MAD |
|---|---|---|---|---|---|---|---|---|---|
| rel | median + 2·MAD | 88.2 / 64.4 / 26.2 | 10.53 | 28.2 | 2 | 10.16 | 0/0/1 | 0/0/1 | 0/0/1 |
| rel | mean + 1σ | 86.6 / 61.6 / 25.7 | 6.18 | 16.6 | 20 | 7.57 | 46/41/0.668 | 63/37/0.012 | 46/31/0.11 |
| rel | 80th percentile | 88.8 / 64.7 / 25.7 | 10.62 | 28.5 | 2 | 11.36 | 23/18/0.533 | 23/26/0.775 | 14/20/0.392 |
| rel | median | 89.0 / 64.1 / 26.1 | 22.28 | 59.7 | 0 | 14.36 | 43/42/1 | 48/45/0.836 | 23/31/0.341 |
| abs | median + 2·MAD | 88.3 / 64.4 / 26.2 | 10.54 | 28.3 | 2 | 10.16 | 0/0/1 | 0/0/1 | 0/0/1 |
| abs | mean + 1σ | 86.7 / 61.7 / 25.8 | 6.18 | 16.6 | 19 | 7.57 | 46/42/0.749 | 62/37/0.0154 | 45/30/0.105 |
| abs | 80th percentile | 88.8 / 64.7 / 25.7 | 10.63 | 28.5 | 2 | 11.36 | 23/18/0.533 | 23/26/0.775 | 14/19/0.487 |
| abs | median | 88.9 / 64.1 / 26.1 | 22.28 | 59.7 | 0 | 14.36 | 43/42/1 | 48/45/0.836 | 24/30/0.497 |

