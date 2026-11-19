#!/usr/bin/env python3
"""
module_timing.py — the two numbers missing from Table 15 (R3-#3).

    python module_timing.py --video path/to/any/UCF-Crime/video.mp4

Measures, on THIS machine:

  A. frame decode + extraction, seconds per extracted frame
     (seek to a timestamp, decode, convert to RGB, resize — exactly what
      the pipeline does before a frame reaches a captioning model)

  B. text normalisation + sentence encoding + 14 cosine similarities,
     seconds per video
     (NLTK tokenise, stop-word filter, WordNet lemmatise, encode with
      all-mpnet-base-v2, score against the 14 descriptors)

Both are the stages Table 15 marks XXX. Takes about a minute.

If you do not have a video file handy, pass --skip-frames and fill only
the text-domain row; the frame-decode row can be dropped from the table
without weakening the argument, since it is common to both VEZA and
E-VEZA and scales identically.
"""
import argparse, time, statistics, sys

N_FRAMES = 20          # frames to decode for the per-frame average
INTERVAL = 3.0         # seconds between extracted frames, as in the paper
N_TEXT_REPEATS = 5     # repeats of the text pipeline, median reported

# a caption document of realistic length: 17 frames x ~25 words,
# which is E-VEZA's mean scheduled frame count at 242 s / 20.7 %
SAMPLE_CAPTION = ("a group of people are standing in a parking lot next to "
                  "a white car while smoke rises in the background ")
DESCRIPTORS = [
    "abuse, maltreatment, mistreatment, cruelty, harm, exploitation, violence, harassment, injury, oppression, torment, child abuse, domestic violence, assault, neglect",
    "arrest, detain, apprehend, capture, take into custody, seize, law enforcement, police, handcuffs, booking, incarceration, detention, custody, officer, suspect, charge, trial",
    "arson, fire-raising, incendiarism, pyromania, torching, fire, blaze, ignition, burning, conflagration, fire-setting, combustibles, smoke, flame, accelerant, burn damage",
    "assault, attack, physical attack, aggression, violence, strike, punch, hit, beating, attack victim, confrontation, battery, fight, injury",
    "burglary, break-in, robbery, housebreaking, theft, intrusion, trespassing, stealing, larceny, heist, pilfering, forced entry, property crime, looting, break-in attempt",
    "explosion, blast, detonation, eruption, blowup, bomb, burst, combustion, detonate, rupture, outburst, shockwave, debris, fireball, incendiary device, explosion site",
    "fighting, brawling, combat, conflict, struggle, tussle, altercation, skirmish, clash, scuffle, fray, melee, fistfight, physical conflict, group fight, confrontation",
    "accident, crash, collision, traffic accident, car accident, vehicle incident, injury, wreck, pileup, road mishap, crash scene, accident site, emergency, unintended event",
    "normal videos, routine, ordinary, standard, typical, regular, everyday, usual, common, mundane, conventional, normalcy, safe, calm, peaceful, non-criminal",
    "robbery, theft, heist, larceny, hold-up, mugging, burglary, stealing, pilfering, banditry, thievery, looting, armed robbery, stick-up, robbery attempt",
    "shooting, gunfire, gunfight, gunplay, firing, discharge, firearm, bullet, shootout, sniper, marksman, ballistics, gun incident, armed attack, shots fired",
    "shoplifting, theft, pilfering, larceny, stealing, shop theft, retail theft, snatching, shop burglarizing, petty theft, retail crime, store theft, concealment, stolen goods",
    "stealing, theft, larceny, robbery, pilfering, thievery, embezzlement, burglary, shoplifting, swiping, looting, filching, property theft, criminal taking",
    "vandalism, destruction, defacement, sabotage, graffiti, wrecking, ruin, property damage, tagging, malicious damage, spray paint, broken windows, destruction of property",
]


def time_frames(path):
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"could not open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = total / fps if fps else 0
    print(f"  video: {dur:.1f}s @ {fps:.1f} fps")

    times = []
    for i in range(N_FRAMES):
        t = (i * INTERVAL) % max(dur - 1.0, 1.0)
        t0 = time.perf_counter()
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        img = Image.fromarray(frame[:, :, ::-1])       # BGR -> RGB
        w, h = img.size                                # mPLUG-2 resize rule
        if max(w, h) > 1024:
            s = 1024.0 / max(w, h)
            img = img.resize((int(w * s), int(h * s)))
        times.append(time.perf_counter() - t0)
    cap.release()
    if not times:
        sys.exit("no frames decoded")
    return statistics.median(times)


def time_text():
    import numpy as np, nltk
    from nltk.corpus import stopwords, wordnet     # noqa: F401
    from nltk.stem import WordNetLemmatizer
    from sentence_transformers import SentenceTransformer

    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    lem = WordNetLemmatizer()
    sw = set(stopwords.words("english"))
    doc = SAMPLE_CAPTION * 17

    # descriptors are encoded once for the whole dataset, not per video
    U = model.encode(DESCRIPTORS, convert_to_numpy=True)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)

    model.encode(["warmup"])                        # exclude lazy init

    times = []
    for _ in range(N_TEXT_REPEATS):
        t0 = time.perf_counter()
        toks = nltk.word_tokenize(doc.lower())
        toks = [lem.lemmatize(w) for w in toks if w.isalpha() and w not in sw]
        v = model.encode(" ".join(toks), convert_to_numpy=True)
        v = v / np.linalg.norm(v)
        _ = U @ v                                   # 14 cosine similarities
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="any UCF-Crime .mp4")
    ap.add_argument("--skip-frames", action="store_true")
    a = ap.parse_args()

    frame_s = None
    if not a.skip_frames:
        if not a.video:
            sys.exit("pass --video <file> or --skip-frames")
        print("[A] frame decode + extraction")
        frame_s = time_frames(a.video)

    print("[B] normalisation + encoding + cosine")
    text_s = time_text()

    print("\n" + "=" * 62)
    print("TABLE 15 — the cells marked XXX")
    print("=" * 62)
    if frame_s is not None:
        print(f"Frame decode + extraction   {frame_s:.3f} s/frame")
        print(f"   VEZA   (81 frames)       {81*frame_s:7.1f} s")
        print(f"   E-VEZA (17 frames)       {17*frame_s:7.1f} s")
    print(f"Normalisation+enc+cosine    {text_s:.3f} s/video   "
          f"(same for both)")

    if frame_s is not None:
        veza = 81 * frame_s + 243.0 + text_s
        eveza = 0.9 + 17 * frame_s + 51.0 + text_s
        print(f"\nTotal, mPLUG-2              VEZA {veza:7.1f} s"
              f"   E-VEZA {eveza:7.1f} s"
              f"   ({100*(1-eveza/veza):.1f}\\,% lower)")
        print("\nIf you use the 969-video means instead (safer -- matches "
              "Table 7):")
        v2 = 36.5 * frame_s + 109.5 + text_s
        e2 = 0.9 + 7.6 * frame_s + 22.7 + text_s
        print(f"   36.5 / 7.6 frames        VEZA {v2:7.1f} s"
              f"   E-VEZA {e2:7.1f} s"
              f"   ({100*(1-e2/v2):.1f}\\,% lower)")

    print("\nRound to one decimal in the table. Do not report more "
          "precision than this.")


if __name__ == "__main__":
    main()