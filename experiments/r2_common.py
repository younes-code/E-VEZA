"""
r2_common.py  --  shared helpers for the MTAP round-2 experiments (E-VEZA).

Everything here mirrors the paper (Sections 4.1 and 4.6) and the repo scripts:
  * captions:  "Video/idx MM:SS:mmm[.jpg] ## caption"   (filter_active_captions.py format)
  * windows:   median + k*MAD over B equal bins of event timestamps, strict '>', merge runs
               (run_temporal_selection.py)
  * frames:    SLACK rule, a grid frame at t is kept if t+1 >= a and t <= b  (Eq. 5, delta = 1 s)
  * text:      lower -> NLTK word_tokenize -> drop English stop-words -> WordNet lemmatise
               -> all-mpnet-base-v2, max_seq_length 512, cosine vs 14 descriptors (Table 4)

BEFORE trusting any new number, run  `python r2_text.py check`  and make sure it reproduces
Table 5 (888 videos). If it does not, adjust the CONFIG flags below until it does.
"""
import glob
import json
import os
import re
from math import sqrt

import numpy as np

# ----------------------------------------------------------------------------------
# CONFIG -- the settings variant_rak.py used. Check them against variant_rak.py once.
# ----------------------------------------------------------------------------------
TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_SEQ_LEN = 512            # results doc: "setting maxlen=512"
STRIP_EOS = False            # variant_rak.py keeps "</s>" (mPLUG-Owl2 EOS) in the caption text
NORMALISE_DESCRIPTORS = False  # True if variant_rak runs nu() on the 14 descriptors too
EMPTY_DOC_IS_MISS = True     # variant_rak.py: hit[empty] = False (miss at every K)
DESCRIPTOR_JOIN = " "         # variant_rak.py encodes " ".join(desc.split(", ")) (commas removed)
SLACK = 1.0                  # onset margin delta, seconds

EXPECTED_TABLE5 = {          # (R@10, R@5, R@1) on 888 videos
    "mplug": {"veza": (88.7, 65.4, 26.8), "eveza": (88.9, 66.0, 27.3)},
    "blip":  {"veza": (91.7, 65.5, 25.1), "eveza": (92.5, 65.7, 30.0)},
    "git":   {"veza": (80.3, 50.3, 20.3), "eveza": (81.1, 52.0, 18.5)},
}

# ----------------------------------------------------------------------------------
# Class descriptors (Table 4) and video-name -> class mapping
# ----------------------------------------------------------------------------------
DESCRIPTORS = {
    "Abuse": "abuse, maltreatment, mistreatment, cruelty, harm, exploitation, violence, harassment, injury, oppression, torment, child abuse, domestic violence, assault, neglect",
    "Arrest": "arrest, detain, apprehend, capture, take into custody, seize, law enforcement, police, handcuffs, booking, incarceration, detention, custody, officer, suspect, charge, trial",
    "Arson": "arson, fire-raising, incendiarism, pyromania, torching, fire, blaze, ignition, burning, conflagration, fire-setting, combustibles, smoke, flame, accelerant, burn damage",
    "Assault": "assault, attack, physical attack, aggression, violence, strike, punch, hit, beating, attack victim, confrontation, battery, fight, injury",
    "Burglary": "burglary, break-in, robbery, housebreaking, theft, intrusion, trespassing, stealing, larceny, heist, pilfering, forced entry, property crime, looting, break-in attempt",
    "Explosion": "explosion, blast, detonation, eruption, blowup, bomb, burst, combustion, detonate, rupture, outburst, shockwave, debris, fireball, incendiary device, explosion site",
    "Fighting": "fighting, brawling, combat, conflict, struggle, tussle, altercation, skirmish, clash, scuffle, fray, melee, fistfight, physical conflict, group fight, confrontation",
    "Accident": "accident, crash, collision, traffic accident, car accident, vehicle incident, injury, wreck, pileup, road mishap, crash scene, accident site, emergency, unintended event",
    "Normal": "normal videos, routine, ordinary, standard, typical, regular, everyday, usual, common, mundane, conventional, normalcy, safe, calm, peaceful, non-criminal",
    "Robbery": "robbery, theft, heist, larceny, hold-up, mugging, burglary, stealing, pilfering, banditry, thievery, looting, armed robbery, stick-up, robbery attempt",
    "Shooting": "shooting, gunfire, gunfight, gunplay, firing, discharge, firearm, bullet, shootout, sniper, marksman, ballistics, gun incident, armed attack, shots fired",
    "Shoplifting": "shoplifting, theft, pilfering, larceny, stealing, shop theft, retail theft, snatching, shop burglarizing, petty theft, retail crime, store theft, concealment, stolen goods",
    "Stealing": "stealing, theft, larceny, robbery, pilfering, thievery, embezzlement, burglary, shoplifting, swiping, looting, filching, property theft, criminal taking",
    "Vandalism": "vandalism, destruction, defacement, sabotage, graffiti, wrecking, ruin, property damage, tagging, malicious damage, spray paint, broken windows, destruction of property",
}
CLASSES = list(DESCRIPTORS.keys())
ANOMALY_CLASSES = [c for c in CLASSES if c != "Normal"]
_ALIASES = {"RoadAccidents": "Accident", "Assualt": "Assault"}


def video_class(video_name):
    prefix = re.match(r"[A-Za-z]+", video_name).group(0)
    return _ALIASES.get(prefix, prefix)


# ----------------------------------------------------------------------------------
# Captions
# ----------------------------------------------------------------------------------
_CAP_RE = re.compile(
    r"^(?P<vid>[^/\s]+)/(?P<idx>\d+)\s+(?P<m>\d+):(?P<s>\d{2}):(?P<ms>\d{3})(?:\.jpg)?\s+##\s?(?P<txt>.*)$")


def load_captions(path):
    """{video: [(t_seconds, frame_idx, caption), ...]} sorted by frame index."""
    caps = {}
    with open(path, encoding="utf-8", errors="ignore") as f:   # as variant_rak.py
        for line in f:
            m = _CAP_RE.match(line.rstrip("\n"))
            if not m:
                continue
            t = int(m["m"]) * 60 + int(m["s"]) + int(m["ms"]) / 1000.0
            txt = m["txt"].strip()
            if STRIP_EOS:
                txt = txt.replace("</s>", "").strip()
            caps.setdefault(m["vid"], []).append((t, int(m["idx"]), txt))
    for v in caps:
        caps[v].sort(key=lambda x: x[1])
    return caps


def sampling_interval(caps_of_video):
    """Median gap (s) between consecutive caption timestamps of one video."""
    ts = sorted(c[0] for c in caps_of_video)
    if len(ts) < 2:
        return float("nan")
    return float(np.median(np.diff(ts)))


# ----------------------------------------------------------------------------------
# Windows
# ----------------------------------------------------------------------------------
def load_windows(path):
    """{video: [(a, b), ...]} from global_active_windows.json; reports duplicates."""
    data = json.load(open(path))
    out, dup = {}, 0
    for d in data:
        v = d["video_name"]
        if v in out:
            dup += 1
        out[v] = [tuple(map(float, w)) for w in (d.get("active_windows") or [])]
    if dup:
        print(f"[WARN] {dup} duplicated video names in {path}")
    return out


def windows_from_counts(counts, edges, k=2.0):
    med = np.median(counts)
    mad = np.median(np.abs(counts - med))
    act = counts > med + k * mad
    wins, start = [], None
    for i, a in enumerate(act):
        if a and start is None:
            start = edges[i]
        if not a and start is not None:
            wins.append((float(start), float(edges[i])))
            start = None
    if start is not None:
        wins.append((float(start), float(edges[len(act)])))
    return wins


def windows_from_npz(npz_path, B=200, k=2.0):
    t = np.load(npz_path)["t"] / 1e6
    counts, edges = np.histogram(t, bins=B)
    return windows_from_counts(counts, edges, k)


def windows_multi_B(npz_path, Bs, k=2.0):
    """Load the event stream ONCE and return {B: windows} (same maths as windows_from_npz)."""
    t = np.load(npz_path)["t"] / 1e6
    out = {}
    for B in Bs:
        counts, edges = np.histogram(t, bins=B)
        out[B] = windows_from_counts(counts, edges, k)
    return out


def windows_cache(npz, vids, Bs, cache_path):
    """{video: {B: windows}} for every video with an npz; cached as JSON so eval lists share it."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = [v for v in vids if v in npz and not all(str(B) in cache.get(v, {}) for B in Bs)]
    import time as _t
    t0 = _t.time()
    for i, v in enumerate(todo):
        try:
            w = windows_multi_B(npz[v], Bs)
            cache.setdefault(v, {}).update({str(B): w[B] for B in Bs})
        except Exception as e:
            print(f"[SKIP] {v}: {e}")
            cache.setdefault(v, {})["error"] = str(e)
        if (i + 1) % 50 == 0 or i + 1 == len(todo):
            print(f"  streams read {i+1}/{len(todo)} ({_t.time()-t0:.0f}s)", flush=True)
            json.dump(cache, open(cache_path, "w"))
    return cache


def find_npz(npz_root):
    """{video: path} for every readable .npz under npz_root."""
    out = {}
    for p in glob.glob(os.path.join(npz_root, "**", "*.npz"), recursive=True):
        if os.path.getsize(p) > 0:
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def keep_slack(t, wins, delta=SLACK):
    return any((t + delta >= a) and (t <= b) for a, b in wins)


def select(caps_of_video, wins):
    return [c for c in caps_of_video if keep_slack(c[0], wins)]


# ----------------------------------------------------------------------------------
# Text stage
# ----------------------------------------------------------------------------------
class TextScorer:
    def __init__(self, device=None):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from sentence_transformers import SentenceTransformer
        for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
        self._tok = nltk.word_tokenize
        self._stop = set(stopwords.words("english"))
        self._lem = WordNetLemmatizer()
        self.model = SentenceTransformer(TEXT_MODEL, device=device)
        self.model.max_seq_length = MAX_SEQ_LEN
        self.tokenizer = self.model.tokenizer
        desc = [DESCRIPTOR_JOIN.join(DESCRIPTORS[c].split(", ")) for c in CLASSES]
        if NORMALISE_DESCRIPTORS:
            desc = [self.normalise(d) for d in desc]
        self.class_emb = self._enc(desc)
        self._cache = {}

    def normalise(self, text):
        toks = self._tok(text.lower())
        return " ".join(self._lem.lemmatize(w) for w in toks if w not in self._stop)

    def _enc(self, texts):
        e = self.model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-12)

    def doc(self, captions):
        return self.normalise(" ".join(c[2] for c in captions))

    def n_tokens(self, doc):
        return len(self.tokenizer(doc, add_special_tokens=True, truncation=False)["input_ids"])

    def embed_docs(self, docs, chunked=False):
        """Embeddings for a list of normalised documents (cached)."""
        todo = [d for d in set(docs) if (d, chunked) not in self._cache]
        if todo and not chunked:
            for d, e in zip(todo, self._enc(todo)):
                self._cache[(d, False)] = e
        if todo and chunked:
            for d in todo:
                self._cache[(d, True)] = self._chunked(d)
        return np.stack([self._cache[(d, chunked)] for d in docs])

    def _chunked(self, doc, budget=None):
        """No truncation: split into <= MAX_SEQ_LEN-2 token chunks on word boundaries, mean-pool."""
        budget = budget or MAX_SEQ_LEN - 2
        words = doc.split()
        if not words:
            return self._enc([""])[0]
        lens = [len(x) for x in self.tokenizer(words, add_special_tokens=False)["input_ids"]]
        chunks, cur, n = [], [], 0
        for w, l in zip(words, lens):
            if cur and n + l > budget:
                chunks.append(" ".join(cur)); cur, n = [], 0
            cur.append(w); n += l
        if cur:
            chunks.append(" ".join(cur))
        e = self._enc(chunks).mean(axis=0)
        return e / max(np.linalg.norm(e), 1e-12)

    def true_rank(self, doc_emb, video_names):
        """1-based rank of the true class among the 14 candidates, per video."""
        sims = doc_emb @ self.class_emb.T
        order = np.argsort(-sims, axis=1)
        idx = np.array([CLASSES.index(video_class(v)) for v in video_names])
        return np.array([int(np.where(order[i] == idx[i])[0][0]) + 1 for i in range(len(idx))])


# ----------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------
def rak(ranks, Ks=(10, 5, 1)):
    ranks = np.asarray(ranks)
    return tuple(round(100.0 * float(np.mean(ranks <= k)), 1) for k in Ks)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def mcnemar_exact(hit_a, hit_b):
    """Exact two-sided McNemar on paired binary outcomes. Returns (b, c, p)."""
    from scipy.stats import binomtest
    hit_a, hit_b = np.asarray(hit_a, bool), np.asarray(hit_b, bool)
    b = int(np.sum(hit_a & ~hit_b))   # A right, B wrong
    c = int(np.sum(~hit_a & hit_b))   # A wrong, B right
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[SAVED] {path}")
