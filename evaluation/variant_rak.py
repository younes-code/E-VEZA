#!/usr/bin/env python3
"""
variant_rak.py v2 — R@K per threshold variant, measured.
    --rule slack   your original caption filter ([t, t+1] overlaps a window)
    --rule strict  the paper's rule (t inside a window)
    --rule both    both, one after the other
Stage 1 must reproduce the published 89 / 66 / 28 or it stops.
"""
import argparse, collections, json, os, re, sys, time, zipfile
import numpy as np

B = 200
TARGET = {10: 89.0, 5: 66.0, 1: 28.0}           # published E-VEZA, mPLUG-2
KS = (1, 5, 10)

CANDIDATES = [
 ("Abuse", "abuse, maltreatment, mistreatment, cruelty, harm, exploitation, violence, harassment, injury, oppression, torment, child abuse, domestic violence, assault, neglect"),
 ("Arrest", "arrest, detain, apprehend, capture, take into custody, seize, law enforcement, police, handcuffs, booking, incarceration, detention, custody, officer, suspect, charge, trial"),
 ("Arson", "arson, fire-raising, incendiarism, pyromania, torching, fire, blaze, ignition, burning, conflagration, fire-setting, combustibles, smoke, flame, accelerant, burn damage"),
 ("Assault", "assault, attack, physical attack, aggression, violence, strike, punch, hit, beating, attack victim, confrontation, battery, fight, injury"),
 ("Burglary", "burglary, break-in, robbery, housebreaking, theft, intrusion, trespassing, stealing, larceny, heist, pilfering, forced entry, property crime, looting, break-in attempt"),
 ("Explosion", "explosion, blast, detonation, eruption, blowup, bomb, burst, combustion, detonate, rupture, outburst, shockwave, debris, fireball, incendiary device, explosion site"),
 ("Fighting", "fighting, brawling, combat, conflict, struggle, tussle, altercation, skirmish, clash, scuffle, fray, melee, fistfight, physical conflict, group fight, confrontation"),
 ("Accident", "accident, crash, collision, traffic accident, car accident, vehicle incident, injury, wreck, pileup, road mishap, crash scene, accident site, emergency, unintended event"),
 ("Normal", "normal videos, routine, ordinary, standard, typical, regular, everyday, usual, common, mundane, conventional, normalcy, safe, calm, peaceful, non-criminal"),
 ("Robbery", "robbery, theft, heist, larceny, hold-up, mugging, burglary, stealing, pilfering, banditry, thievery, looting, armed robbery, stick-up, robbery attempt"),
 ("Shooting", "shooting, gunfire, gunfight, gunplay, firing, discharge, firearm, bullet, shootout, sniper, marksman, ballistics, gun incident, armed attack, shots fired"),
 ("Shoplifting", "shoplifting, theft, pilfering, larceny, stealing, shop theft, retail theft, snatching, shop burglarizing, petty theft, retail crime, store theft, concealment, stolen goods"),
 ("Stealing", "stealing, theft, larceny, robbery, pilfering, thievery, embezzlement, burglary, shoplifting, swiping, looting, filching, property theft, criminal taking"),
 ("Vandalism", "vandalism, destruction, defacement, sabotage, graffiti, wrecking, ruin, property damage, tagging, malicious damage, spray paint, broken windows, destruction of property"),
]
CAND_NAMES = [c for c, _ in CANDIDATES]
CLASS_TO_CAND = {"RoadAccidents": "Accident"}

LINE = re.compile(r"^(.+?)/\d+\s+(\d{1,3}:\d{2}:\d{3})(?:\.jpg)?\s+##\s?(.*)$")
NAME_CLASS = re.compile(r"^([A-Za-z]+?)\d+_x264")
RULE = "slack"


def ts(s):
    m, sec, ms = s.split(":")
    return int(m) * 60 + int(sec) + int(ms) / 1000.0


def load_captions(path):
    caps = collections.defaultdict(list)
    last = None
    n_match = n_cont = n_skip = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            m = LINE.match(line)
            if m:
                v, t, c = m.group(1), ts(m.group(2)), m.group(3).strip()
                caps[v].append([t, c])
                last = caps[v][-1]
                n_match += 1
            elif line.strip() and last is not None:
                last[1] += " " + line.strip()
                n_cont += 1
            else:
                n_skip += 1
    for v in caps:
        caps[v].sort(key=lambda x: x[0])
    return caps, n_match, n_cont, n_skip


def kept(caps, windows):
    """slack  = your filter script: caption covers [t, t+1], kept if it
                overlaps any window.
       strict = the paper's rule: kept only if t lies inside a window."""
    if windows is None:
        return [c for _, c in caps]
    out = []
    for t, c in caps:
        for s, e in windows:
            if (RULE == "strict" and s <= t <= e) or \
               (RULE != "strict" and t + 1.0 >= s and t <= e):
                out.append(c)
                break
    return out


class Encoder:
    def __init__(self):
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from sentence_transformers import SentenceTransformer
        home = os.path.expanduser("~/nltk_data")
        os.makedirs(home, exist_ok=True)
        if home not in nltk.data.path:
            nltk.data.path.insert(0, home)
        for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
            try:
                nltk.download(pkg, quiet=True, download_dir=home)
            except Exception:
                pass
        self.nltk = nltk
        self.sw = set(stopwords.words("english"))
        self.lem = WordNetLemmatizer()
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self._desc = {}

    def norm(self, text, alpha):
        toks = self.nltk.word_tokenize(text.lower())
        if alpha:
            toks = [w for w in toks if w.isalpha()]
        return " ".join(self.lem.lemmatize(w) for w in toks if w not in self.sw)

    def docs(self, texts, maxlen, alpha):
        self.model.max_seq_length = maxlen
        normed = [self.norm(t, alpha) if t else "" for t in texts]
        V = self.model.encode(normed, batch_size=32, convert_to_numpy=True,
                              show_progress_bar=False)
        return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    def desc(self, join, maxlen):
        key = (join, maxlen)
        if key not in self._desc:
            self.model.max_seq_length = maxlen
            txt = [join.join(t.split(", ")) for _, t in CANDIDATES]
            U = self.model.encode(txt, convert_to_numpy=True,
                                  show_progress_bar=False)
            self._desc[key] = U / np.linalg.norm(U, axis=1, keepdims=True)
        return self._desc[key]


def rak(V, U, labels, empty):
    order = np.argsort(-(V @ U.T), axis=1)
    out = {}
    for K in KS:
        hit = np.array([labels[i] in order[i, :K] for i in range(len(labels))])
        hit[empty] = False
        out[K] = 100.0 * hit.mean()
    return out


def fmt(r):
    return f"{r[10]:5.1f} {r[5]:5.1f} {r[1]:5.1f}"


def err(r):
    return sum(abs(round(r[k]) - TARGET[k]) for k in KS)


def thresholds(C):
    med = float(np.median(C))
    mad = float(np.median(np.abs(C - med)))
    return {"MAD": med + 2.0 * mad, "Mean": float(C.mean() + C.std()),
            "Fixed": float(np.percentile(C, 80)), "Median": med}


def windows_from(C, thr, edges):
    active = np.flatnonzero(C > thr)
    if active.size == 0:
        return []
    out, start, prev = [], active[0], active[0]
    for b in active[1:]:
        if b == prev + 1:
            prev = b
            continue
        out.append((float(edges[start]), float(edges[prev + 1])))
        start = prev = b
    out.append((float(edges[start]), float(edges[prev + 1])))
    return out


def find_npz(base, cls, name):
    for c in (cls, cls.replace("RoadAccidents", "Road_Accidents"),
              cls.replace("RoadAccidents", "RoadAccident"), cls.lower()):
        p = os.path.join(base, c, name + ".npz")
        if os.path.exists(p) and zipfile.is_zipfile(p):
            return p
    return None


def load_counts(videos, classes, base, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        got = dict(zip(d["names"].tolist(), zip(d["counts"], d["T"])))
        print(f"  histogram cache: {len(got)} videos from {cache}")
        return got
    got, t0 = {}, time.time()
    for i, (v, c) in enumerate(zip(videos, classes)):
        p = find_npz(base, c, v)
        if p is None:
            continue
        try:
            t = np.load(p)["t"] / 1e6
        except Exception:
            continue
        if t.size == 0 or t.max() <= t.min():
            continue
        C, _ = np.histogram(t, bins=B)
        got[v] = (C.astype(float), float(t.max() - t.min()))
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(videos)} streams read "
                  f"({time.time()-t0:.0f}s)", flush=True)
    names = list(got)
    np.savez(cache, names=np.array(names, dtype=object),
             counts=np.stack([got[n][0] for n in names]),
             T=np.array([got[n][1] for n in names]))
    print(f"  cached {len(got)} histograms -> {cache}")
    return got


def stage2(a, enc, caps, pub, videos, classes, labels, maxlen, alpha, join):
    global RULE
    for RULE in (["slack", "strict"] if a.rule == "both" else [a.rule]):
        print("\n" + "=" * 76)
        print(f"STAGE 2 — variants, RULE={RULE}, maxlen={maxlen} "
              f"alpha={alpha} join={join!r}")
        print("=" * 76)
        got = load_counts(videos, classes, a.npz, a.cache)
        idx = [i for i, v in enumerate(videos) if v in got]
        print(f"  videos with a readable event stream: {len(idx)}/{len(videos)}\n")
        sub_v = [videos[i] for i in idx]
        sub_l = labels[idx]
        U = enc.desc(join, maxlen)

        rows = []
        ftxt = [" ".join(kept(caps[v], None)) for v in sub_v]
        rows.append(("Full video (VEZA)", rak(enc.docs(ftxt, maxlen, alpha), U, sub_l, np.zeros(len(sub_v), bool)), 0, float(np.mean([len(caps[v]) for v in sub_v]))))
        txt = [" ".join(kept(caps[v], pub[v])) for v in sub_v]
        empty = np.array([not t for t in txt])
        rows.append(("Published windows",
                     rak(enc.docs(txt, maxlen, alpha), U, sub_l, empty),
                     empty.sum(),
                     np.mean([len(kept(caps[v], pub[v])) for v in sub_v])))
        for var in (() if a.pub_only else ("MAD", "Mean", "Fixed", "Median")):
            wins = {}
            for v in sub_v:
                C, T = got[v]
                wins[v] = windows_from(C, thresholds(C)[var],
                                       np.linspace(0.0, T, B + 1))
            txt = [" ".join(kept(caps[v], wins[v])) for v in sub_v]
            empty = np.array([not t for t in txt])
            rows.append((var,
                         rak(enc.docs(txt, maxlen, alpha), U, sub_l, empty),
                         empty.sum(),
                         np.mean([len(kept(caps[v], wins[v])) for v in sub_v])))
            print(f"  {var:<8} done", flush=True)

        print(f"\n{'Variant':<20}{'R@10':>7}{'R@5':>7}{'R@1':>7}"
              f"{'no caption':>12}{'captions/vid':>14}")
        print("-" * 76)
        for name, r, ne, cpv in rows:
            print(f"{name:<20}{r[10]:>7.1f}{r[5]:>7.1f}{r[1]:>7.1f}"
                  f"{int(ne):>12}{cpv:>14.1f}")
        print(f"\nN = {len(sub_v)} videos.")


def main():
    global RULE
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--npz")
    ap.add_argument("--cache", default="variant_counts_cache.npz")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rule", choices=["slack", "strict", "both"], default="slack")
    ap.add_argument("--pub-only", action="store_true")
    ap.add_argument("--setting", default=None,
                    help="skip stage 1, e.g. '512,True, '")
    a = ap.parse_args()

    caps, nm, nc, ns = load_captions(a.captions)
    print(f"captions : {nm} caption lines, {nc} continuation lines, "
          f"{ns} other; {len(caps)} videos")
    if nm == 0:
        sys.exit("no caption line matched")

    recs = json.load(open(a.json, encoding="utf-8"))
    pub = {r["video_name"]: r.get("active_windows", []) for r in recs}

    videos, classes, labels = [], [], []
    for v in sorted(pub):
        if v not in caps:
            continue
        m = NAME_CLASS.match(v)
        if not m:
            continue
        cls = m.group(1)
        cand = CLASS_TO_CAND.get(cls, cls)
        if cand not in CAND_NAMES:
            continue
        videos.append(v)
        classes.append(cls)
        labels.append(CAND_NAMES.index(cand))
    labels = np.array(labels)
    print(f"evaluated: {len(videos)} videos in both the JSON and the captions\n")

    enc = Encoder()

    if a.setting:
        ml, al, jn = a.setting.split(",", 2)
        return stage2(a, enc, caps, pub, videos, classes, labels,
                      int(ml), al.strip() == "True", jn)

    # ── STAGE 1: reproduction (always with your original filter) ──
    RULE = "slack"
    print("=" * 76)
    print("STAGE 1 — reproduction. target R@10 89 / R@5 66 / R@1 28")
    print("=" * 76)
    full_txt = [" ".join(kept(caps[v], None)) for v in videos]
    pub_txt = [" ".join(kept(caps[v], pub[v])) for v in videos]
    pub_empty = np.array([not t for t in pub_txt])
    print(f"published windows: mean "
          f"{np.mean([len(kept(caps[v], pub[v])) for v in videos]):.1f} "
          f"captions/video (full: {np.mean([len(caps[v]) for v in videos]):.1f})\n")
    print(f"{'maxlen':>6} {'alpha':>6} {'join':>5} | {'FULL R@10 R@5 R@1':>18} "
          f"| {'PUBLISHED WINDOWS':>18} | err")
    print("-" * 76)
    grid = []
    for maxlen in (384, 512):
        for alpha in (True, False):
            t0 = time.time()
            Vf = enc.docs(full_txt, maxlen, alpha)
            Vp = enc.docs(pub_txt, maxlen, alpha)
            for join in (" ", ", "):
                U = enc.desc(join, maxlen)
                rf = rak(Vf, U, labels, np.zeros(len(labels), bool))
                rp = rak(Vp, U, labels, pub_empty)
                grid.append((err(rp), maxlen, alpha, join, rf, rp))
                print(f"{maxlen:>6} {str(alpha):>6} {repr(join):>5} | "
                      f"   {fmt(rf)} |    {fmt(rp)} | {err(rp)}", flush=True)
            print(f"{'':>20}({time.time()-t0:.0f}s)")

    grid.sort(key=lambda x: x[0])
    e, maxlen, alpha, join, rf, rp = grid[0]
    print(f"\nbest setting: maxlen={maxlen} alpha={alpha} join={join!r} "
          f"-> {fmt(rp)} (error {e})")
    if e > 2 and not a.force:
        print("\n!! NOTHING REPRODUCES 89/66/28 within 2 points. Stopping.")
        return
    print("REPRODUCED." if e <= 2 else "!! forced past a failed reproduction")
    if not a.npz:
        return
    print(f"\n>>> rerun stage 2 alone later with:  --setting '{maxlen},{alpha},{join}'")
    stage2(a, enc, caps, pub, videos, classes, labels, maxlen, alpha, join)


if __name__ == "__main__":
    main()