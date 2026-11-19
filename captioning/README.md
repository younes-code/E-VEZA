# Caption generation

These scripts caption every frame of VEZA's 3 s sampling grid. The resulting files are in
`outputs/captions/full/`, one line per frame:

```
<video>/<index> MM:SS:mmm[.jpg] ## <caption>
```

E-VEZA does not re-caption anything. `gate/filter_active_captions.py` keeps the lines whose
timestamp falls in an active window.

| Backbone | Script | Checkpoint used for the released captions | Settings |
|---|---|---|---|
| mPLUG-Owl2 | `mplug_owl2/caption_generation.py` | `MAGAer13/mplug-owl2-llama2-7b` (FP16) | sampling, temperature 0.7, max 512 new tokens |
| BLIP-2 | `blip2/generate_captions.py` | `Salesforce/blip2-opt-2.7b` (8-bit) | beam search (5 beams), max length 1000 |
| GIT | `git/caption_generation.py` | `microsoft/git-base-coco` | see script |

Notes on provenance:

* **mPLUG-Owl2** needs the official repository (`X-PLUG/mPLUG-Owl`, folder `mPLUG-Owl2`) on the
  Python path; the script imports its `mplug_owl2` package. The VEZA paper refers to this model
  as mPLUG-2.
* **BLIP-2**: the 8-bit, 5-beam configuration above is the one that produced
  `outputs/captions/full/blip_captions.txt`. BLIP-2 was run on a 1 s grid for most videos;
  Section 4.6 of the paper and `results/blip_sampling_by_class.txt` give the details.
* **GIT**: the released captions were produced by `microsoft/git-base-coco`, the script's
  default. Regenerating 12 frames with it reproduces all 12 stored captions
  exactly (`experiments/git_provenance.py`, `results/git_provenance.json`).

Each script takes `--frames-dir` (extracted frames, one sub-folder per video) and `--out`
(caption file, appended to).
