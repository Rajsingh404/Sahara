"""Evaluate and export name-call detection on the team's "Aryan" recordings.

Input: 16 kHz mono WAVs decoded from the organised set (data/raw/personalization/aryan/ and
data/raw/indian_ambient/background/), e.g.

    for f in data/raw/personalization/aryan/*/*.m4a data/raw/indian_ambient/background/*; do
      ffmpeg -loglevel error -i "$f" -ac 1 -ar 16000 "data/interim/name_wav/$(basename "${f%.*}").wav"
    done
    python scripts/eval_name_call.py --wav-dir data/interim/name_wav --report docs/name_call_eval.md \
        --export app/assets

Questions answered, each the way the phone faces them:

1. Is "Aryan" spotted when the caller's voice was never in training? Leave one recorder out.
2. Who is calling, when every caller is enrolled? Leave one clip out.
3. Is an unenrolled caller reported as "Someone" instead of a wrong name? Leave one caller out.

--tts-dir adds synthetic clips (scripts/make_tts_name_clips.py) to training only; every number is still measured
on the team's real recordings.

--export trains on everything and writes name_head.json (and speaker_encoder.bin when
resemblyzer is installed) for the app.
"""
from __future__ import annotations

import argparse
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.personalization.name_head import mix, speech_bounds, train_head  # noqa: E402
from src.personalization.speech_embedding import SpeechEmbedder  # noqa: E402

TTS_RE = re.compile(r"tts_(pos|neg)__([a-z]+)__\d+")
NAME_RE = re.compile(r"aryan_(pos|neg)__([a-z]+)__\d+")
BG_RE = re.compile(r"background_([a-z_]+)__([a-z]+)__\d+")
CHUNK = 48_000  # background scenes are scored in 3 s pieces, like live windows


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != 16_000 or w.getnchannels() != 1:
            raise SystemExit(f"{path}: decode to 16 kHz mono first (ffmpeg -ac 1 -ar 16000)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def roc_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a positive scores above a negative)."""
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """(equal error rate, threshold): where missed calls and false calls are closest."""
    best_err, best_thr = 1.0, 0.5
    for t in np.unique(np.concatenate([pos, neg])):
        err = max(float((pos < t).mean()), float((neg >= t).mean()))
        if err < best_err:
            best_err, best_thr = err, float(t)
    return best_err, best_thr


def load(wav_dir: Path, embedder: SpeechEmbedder):
    clips, scenes = [], []
    for p in sorted(wav_dir.glob("*.wav")):
        if m := NAME_RE.match(p.stem):
            w = read_wav(p)
            a, b = speech_bounds(w)
            clips.append({"id": p.stem, "pos": m.group(1) == "pos", "who": m.group(2), "wave": w,
                          "emb": embedder.embed(w), "core": w[a:b]})
        elif m := BG_RE.match(p.stem):
            w = read_wav(p)
            for k, i in enumerate(range(0, max(1, len(w) - CHUNK + 1), CHUNK)):
                scenes.append({"id": f"{p.stem}#{k}", "scene": m.group(1), "who": m.group(2), "wave": w[i : i + CHUNK]})
    for s in scenes:
        s["emb"] = embedder.embed(s["wave"])
    return clips, scenes


def load_tts(tts_dir: Path | None, scenes, embedder, rng) -> tuple[list, list]:
    """Synthetic clips as (embedding, recorder of the mixed-in scene or None): clean, plus one copy over a background
    scene. The recorder tag lets each fold drop copies made with the held-out recorder's scenes."""
    pos, neg = [], []
    if tts_dir is None:
        return pos, neg
    for p in sorted(tts_dir.glob("tts_*.wav")):
        if m := TTS_RE.match(p.stem):
            w = read_wav(p)
            a, b = speech_bounds(w)
            core = w[a:b]
            scene = scenes[rng.integers(len(scenes))]
            noisy = mix(core, scene["wave"], rng.uniform(5, 20), rng)
            (pos if m.group(1) == "pos" else neg).extend([(embedder.embed(core), None), (embedder.embed(noisy), scene["who"])])
    return pos, neg


def augmented_positives(clips, scenes, embedder, rng, per_clip: int):
    out = []
    for c in clips:
        out.append(embedder.embed(c["core"]))
        for _ in range(per_clip):
            noise = scenes[rng.integers(len(scenes))]["wave"]
            out.append(embedder.embed(mix(c["core"], noise, rng.uniform(5, 20), rng)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav-dir", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, help="openWakeWord models (default: the installed wheel)")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--export", type=Path, help="write name_head.json (+ speaker_encoder.bin) here")
    ap.add_argument("--augment", type=int, default=3, help="noisy copies of each positive (default 3)")
    ap.add_argument("--tts-dir", type=Path, help="synthetic clips for training only (make_tts_name_clips.py)")
    ap.add_argument("--tts-use", choices=("both", "pos", "neg"), default="both",
                    help="which synthetic clips to train on (default both)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    embedder = SpeechEmbedder(args.models_dir)
    clips, scenes = load(args.wav_dir, embedder)
    pos = [c for c in clips if c["pos"]]
    recorders = sorted({c["who"] for c in clips})
    rng = np.random.default_rng(args.seed)
    tts_pos, tts_neg = load_tts(args.tts_dir, scenes, embedder, rng)
    tts_pos = tts_pos if args.tts_use in ("both", "pos") else []
    tts_neg = tts_neg if args.tts_use in ("both", "neg") else []
    lines = [
        "# Name-call evaluation", "",
        f"{len(pos)} \"Aryan\" clips, {len(clips) - len(pos)} similar-name clips and {len(scenes)} three-second background "
        f"pieces (TV, kitchen, hawkers, temple, rain, door and drawer) from recorders {', '.join(recorders)}. "
        "Generated by `scripts/eval_name_call.py`.", "",
    ]
    if tts_pos or tts_neg:
        lines += [f"Training also used {len(tts_pos) // 2} synthetic \"Aryan\" and {len(tts_neg) // 2} synthetic similar-name "
                  "clips from the macOS voices (Indian English, Hindi and others; `scripts/make_tts_name_clips.py`), each "
                  "clean and mixed with background. They are never scored: every number below is on the team's real "
                  "recordings.", ""]

    # 1. Spotting, leave one recorder out. Background scenes are split by recorder too.
    scores = defaultdict(list)
    for r in recorders:
        train_pos = [c for c in pos if c["who"] != r]
        train_neg = [c["emb"] for c in clips if not c["pos"] and c["who"] != r] + [s["emb"] for s in scenes if s["who"] != r]
        head = train_head("aryan", augmented_positives(train_pos, [s for s in scenes if s["who"] != r], embedder, rng, args.augment)
                          + [e for e, w in tts_pos if w != r], train_neg + [e for e, w in tts_neg if w != r])
        for c in clips:
            if c["who"] == r:
                scores["pos" if c["pos"] else "similar"].append(head.score_clip(c["emb"]))
        for s in scenes:
            if s["who"] == r:
                scores["scene"].append(head.score_clip(s["emb"]))
                scores[f"scene:{s['scene']}"].append(head.score_clip(s["emb"]))
    p = np.array(scores["pos"])
    sim, scn = np.array(scores["similar"]), np.array(scores["scene"])
    allneg = np.concatenate([sim, scn])
    e_all, thr = eer(p, allneg)
    lines += [
        "## 1. Spotting \"Aryan\" from a voice the model never heard", "",
        "Leave one recorder out: the head is trained on the other three recorders' clips and scenes, then scored on the fourth.", "",
        "| Negatives | ROC AUC | Equal error rate |", "|---|---:|---:|",
        f"| Background scenes (TV, kitchen, street...) | {roc_auc(p, scn):.3f} | {eer(p, scn)[0]:.0%} |",
        f"| Similar names (Arya, Aryanshi, other names) | {roc_auc(p, sim):.3f} | {eer(p, sim)[0]:.0%} |",
        f"| Both | {roc_auc(p, allneg):.3f} | {e_all:.0%} |", "",
        f"At the equal-error threshold ({thr:.2f}): {(p >= thr).mean():.0%} of calls caught, "
        f"{(scn >= thr).mean():.0%} of background pieces and {(sim >= thr).mean():.0%} of similar names raise a false call.", "",
        "False calls per background scene at that threshold:", "",
        "| Scene | Pieces | False calls |", "|---|---:|---:|",
    ]
    for key in sorted(k for k in scores if k.startswith("scene:")):
        v = np.array(scores[key])
        lines.append(f"| {key[6:]} | {len(v)} | {(v >= thr).sum()} |")

    # 2-3. Caller identity.
    try:
        from src.personalization.speaker import SpeakerEncoder
    except ImportError as exc:
        lines += ["", "## 2-3. Who is calling", "", f"Skipped: {exc}. `pip install resemblyzer` to run it."]
        encoder = None
    else:
        encoder = SpeakerEncoder()
        lines += ["", *caller_section(clips, recorders, encoder)]

    lines += [
        "", "## Caveats", "",
        "- Each recorder used their own phone and room, so caller identity here is partly phone and room identity. "
        "Re-run after the family enrols on Aryan's own phone, which is how the app is used.",
        "- \"Recorder\" stands in for \"caller\": the clips do not say who spoke. If a recorder filmed someone else, "
        "that clip is counted under the recorder.",
        "- The spotter is trained for one name. Another user's name needs its own clips (or synthetic TTS clips) and a retrain.",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.report:
        args.report.write_text(text, encoding="utf-8")

    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        head = train_head("aryan", augmented_positives(pos, scenes, embedder, rng, args.augment) + [e for e, _ in tts_pos],
                          [c["emb"] for c in clips if not c["pos"]] + [s["emb"] for s in scenes] + [e for e, _ in tts_neg])
        head.threshold = thr
        head.to_json(args.export / "name_head.json", {"display_name": "Aryan", "eer_threshold_loro": thr})
        print(f"wrote {args.export / 'name_head.json'} (threshold {thr:.2f})")
        if encoder is not None:
            encoder.export(args.export / "speaker_encoder.bin")
            print(f"wrote {args.export / 'speaker_encoder.bin'}")


def caller_section(clips, recorders, encoder) -> list[str]:
    for c in clips:
        c["voice"] = encoder.embed(c["wave"])
    pos = [c for c in clips if c["pos"]]
    by_who = defaultdict(list)
    for c in pos:
        by_who[c["who"]].append(c)

    def centroid(vs):
        m = np.mean([v / np.linalg.norm(v) for v in vs], axis=0)
        return m / np.linalg.norm(m)

    confusion = defaultdict(lambda: defaultdict(int))
    genuine = []
    for c in pos:
        cents = {w: centroid([x["voice"] for x in cs if x["id"] != c["id"]]) for w, cs in by_who.items()}
        sims = {w: float(c["voice"] @ v / np.linalg.norm(c["voice"])) for w, v in cents.items()}
        guess = max(sims, key=sims.get)
        confusion[c["who"]][guess] += 1
        genuine.append((guess == c["who"], sims[guess]))
    correct = sum(g for g, _ in genuine)
    impostor = []
    for r in recorders:
        cents = {w: centroid([x["voice"] for x in cs]) for w, cs in by_who.items() if w != r}
        for c in by_who[r]:
            impostor.append(max(float(c["voice"] @ v / np.linalg.norm(c["voice"])) for v in cents.values()))
    impostor = np.array(impostor)
    out = [
        "## 2. Who is calling (every caller enrolled)", "",
        f"Leave one clip out, the rest enrol each caller: **{correct}/{len(pos)} = {correct / len(pos):.0%}** named correctly.", "",
        "| Said by \\ named as | " + " | ".join(recorders) + " |", "|---" * (len(recorders) + 1) + "|",
    ]
    for r in recorders:
        out.append(f"| {r} | " + " | ".join(str(confusion[r][g]) for g in recorders) + " |")
    out += [
        "", "## 3. A caller who is not enrolled", "",
        "Leave one caller out entirely. Below the accept threshold the app says \"Someone is calling you\". "
        "Naming the wrong person is worse than \"Someone\" (the user still learns someone is calling), so the app "
        "defaults to 0.70.", "",
        "| Accept threshold | Enrolled callers named correctly | Unenrolled callers shown as \"Someone\" |", "|---:|---:|---:|",
    ]
    for t in (0.60, 0.65, 0.70, 0.75, 0.80):
        named = sum(1 for g, s in genuine if g and s >= t)
        out.append(f"| {t:.2f} | {named}/{len(pos)} | {(impostor < t).mean():.0%} |")
    return out


if __name__ == "__main__":
    main()
