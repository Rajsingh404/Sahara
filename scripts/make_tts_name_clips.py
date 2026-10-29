"""Generate synthetic "Aryan" and similar-name clips with the macOS voices, for name-call training only.

The team recorded 38 "Aryan" clips and 19 similar-name clips. These synthetic clips add many more voices and,
above all, more near-miss names (Arya, Aryanshi, Ayaan...), which are the spotter's weakest case. They are used
only to train the head; every reported number is still measured on the team's real recordings.

Result (docs/name_call.md): every variant scored worse than real recordings alone, so the shipped head does not
use these clips. Kept so the experiment can be repeated, e.g. with better voices.

    python scripts/make_tts_name_clips.py --out data/interim/name_tts
    python scripts/eval_name_call.py --wav-dir data/interim/name_wav --tts-dir data/interim/name_tts ...

Output names: tts_pos__<voice>__<n>.wav / tts_neg__<voice>__<n>.wav, 16 kHz mono. Audio stays local (data/ is not
committed). macOS only (`say`); needs ffmpeg.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

# Human-sounding English voices that ship with macOS, plus the Hindi one. Novelty voices are left out.
VOICES = [
    "Aman", "Rishi (English (India))", "Lekha", "Daniel", "Karen", "Moira", "Samantha", "Tessa", "Tara", "Fred",
    "Kathy", "Ralph", "Junior", "Eddy (English (UK))", "Eddy (English (US))", "Flo (English (UK))",
    "Flo (English (US))", "Reed (English (UK))", "Reed (English (US))", "Rocko (English (UK))", "Rocko (English (US))",
    "Sandy (English (UK))", "Sandy (English (US))", "Shelley (English (UK))", "Shelley (English (US))",
    "Grandma (English (UK))", "Grandpa (English (UK))",
]
POSITIVE = ["Aryan", "Aryan!", "Aryan?", "Aryan, come here", "Aryan, listen"]
# Near misses first: these are what the head confuses today.
NEGATIVE = ["Arya", "Aryanshi", "Ayaan", "Aryav", "Rayan", "Ryan", "Aran", "Arjun", "Ayan", "Arun", "Karan",
            "Aman", "Adrian", "Iran", "Are you", "Arya, come here", "Ayaan, listen"]
HINDI = {"Aryan": "आर्यन", "Aryan!": "आर्यन!", "Aryan?": "आर्यन?", "Aryan, come here": "आर्यन, इधर आओ",
         "Aryan, listen": "आर्यन, सुनो", "Arya": "आर्या", "Aryanshi": "आर्यांशी", "Ayaan": "अयान", "Aryav": "आरव",
         "Rayan": "रायन", "Ryan": "रायन", "Aran": "अरन", "Arjun": "अर्जुन", "Ayan": "अयन", "Arun": "अरुण",
         "Karan": "करण", "Aman": "अमन", "Adrian": "एड्रियन", "Iran": "ईरान", "Are you": "आर यू",
         "Arya, come here": "आर्या, इधर आओ", "Ayaan, listen": "अयान, सुनो"}
RATES = [140, 180, 220]  # words per minute: slow call, normal, quick


def slug(voice: str) -> str:
    return re.sub(r"[^a-z]+", "", voice.lower().replace("english", ""))


def speak(voice: str, text: str, rate: int, out: Path, tmp: Path) -> bool:
    if out.exists():
        return True
    aiff = tmp / "x.aiff"
    aiff.unlink(missing_ok=True)
    try:  # `say` occasionally hangs on a word; skip it rather than stall the run
        r = subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), text],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode or not aiff.exists():
        return False
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(aiff), "-ac", "1", "-ar", "16000", str(out)],
                   check=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/interim/name_tts"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {"pos": 0, "neg": 0}
    with tempfile.TemporaryDirectory() as t:
        for voice in VOICES:
            n = 0
            for kind, texts in (("pos", POSITIVE), ("neg", NEGATIVE)):
                for text in texts:
                    spoken = HINDI.get(text, text) if voice == "Lekha" else text
                    for rate in RATES:
                        n += 1
                        if speak(voice, spoken, rate, args.out / f"tts_{kind}__{slug(voice)}__{n:03d}.wav", Path(t)):
                            counts[kind] += 1
            print(f"{voice}: done")
    print(f"wrote {counts['pos']} positive and {counts['neg']} negative clips to {args.out}")


if __name__ == "__main__":
    main()
