"""End-to-end check: replay test clips through exactly what the app runs, using only app/assets/.

Per clip: 16-bit PCM (as the microphone delivers) -> 1 s ring buffer, 0.5 s hops -> VAD on the
newest hop -> yamnet.tflite on the newest 15,600 samples -> embedding pool (layout chosen from the
head's input size, as InferenceEngine.kt does) -> sahara_classifier.tflite -> DetectionPolicy with
the bundled thresholds and hits. Compared with the Python evaluation pipeline (hop cache -> Keras
head on CPU -> same policy): max score difference and whether the same alerts fire.

    python app/tool/check_app_pipeline.py [--clips 60]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.config import (  # noqa: E402
    HOP_SAMPLES,
    METADATA_DIR,
    SOUND_CLASSES,
    VAD_ENERGY_THRESHOLD,
    WINDOW_SAMPLES,
    YAMNET_PATCH_SAMPLES,
)
from src.evaluation.policy import DetectionPolicy  # noqa: E402
from src.evaluation.streaming import clip_hop_scores, load_head, selected_run_dir  # noqa: E402
from src.preprocessing.audio_utils import load_audio  # noqa: E402

ASSETS = REPO_ROOT / "app" / "assets"


class TfliteModel:
    def __init__(self, path: Path, output_last_dim: int | None = None):
        self.it = tf.lite.Interpreter(model_path=str(path))
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        outs = self.it.get_output_details()
        self.out = next(o for o in outs if output_last_dim is None or o["shape"][-1] == output_last_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.it.set_tensor(self.inp["index"], x.reshape(self.inp["shape"]).astype(np.float32))
        self.it.invoke()
        return self.it.get_tensor(self.out["index"]).reshape(-1)


def app_scores(pcm16: np.ndarray, yamnet: TfliteModel, head: TfliteModel, pools: tuple[int, ...]) -> np.ndarray:
    """Per-hop scores exactly as AudioCaptureService.kt + InferenceEngine.kt produce them."""
    ring = np.zeros(WINDOW_SAMPLES, np.float32)
    filled, pool = 0, deque(maxlen=max(pools))
    out = []
    for h in range(len(pcm16) // HOP_SAMPLES):
        ring = np.concatenate([ring[HOP_SAMPLES:], pcm16[h * HOP_SAMPLES:(h + 1) * HOP_SAMPLES].astype(np.float32) / 32768.0])
        filled = min(filled + HOP_SAMPLES, WINDOW_SAMPLES)
        if filled < WINDOW_SAMPLES:
            continue
        if float(np.mean(ring[-HOP_SAMPLES:].astype(np.float64) ** 2)) < VAD_ENERGY_THRESHOLD:
            out.append(np.zeros(len(SOUND_CLASSES), np.float32))  # silent hop: zero scores to the policy
            continue
        pool.append(yamnet(ring[-YAMNET_PATCH_SAMPLES:]))
        items = list(pool)
        vec = np.concatenate([np.mean(items[-p:], axis=0) for p in pools])
        out.append(head(vec))
    return np.asarray(out, np.float32).reshape(-1, len(SOUND_CLASSES))


def alerts(scores: np.ndarray, thresholds: dict, hits: dict) -> list[str]:
    policy = DetectionPolicy(SOUND_CLASSES, thresholds, hits_per_class=hits)
    fired = []
    for i, row in enumerate(scores):
        fired += [d.label for d in policy.update(row, i * 500)]
    return sorted(set(fired))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", type=int, default=60)
    args = parser.parse_args()

    cfg = json.loads((ASSETS / "class_thresholds.json").read_text())
    thresholds = {c: cfg[c]["threshold"] for c in SOUND_CLASSES}
    hits = {c: cfg[c]["hits_required"] for c in SOUND_CLASSES if "hits_required" in cfg[c]}
    labels = [ln for ln in (ASSETS / "labels.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    assert labels == list(SOUND_CLASSES), "labels.txt does not match SOUND_CLASSES"
    yamnet = TfliteModel(ASSETS / "yamnet.tflite", output_last_dim=1024)
    head = TfliteModel(ASSETS / "sahara_classifier.tflite")
    head_in = int(np.prod(head.inp["shape"]))
    pools = {1024: (2,), 2048: (2, 6)}[head_in]
    keras_head = load_head(selected_run_dir(cfg["_meta"]["run"]))

    with (METADATA_DIR / "test_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["source_type"] != "synthetic_mixed"]
    by_label: dict[str, list[str]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r["filepath"])
    rng = random.Random(0)
    per = max(1, args.clips // (len(SOUND_CLASSES) + 1))
    picked = sorted({fp for lbl in sorted(by_label) for fp in rng.sample(by_label[lbl], min(per, len(by_label[lbl])))})

    worst, same_alerts, compared = 0.0, 0, 0
    for fp in picked:
        wave = load_audio(fp, peak_normalize=False)
        if wave is None:
            continue
        pcm16 = np.clip(np.round(wave * 32767), -32768, 32767).astype(np.int16)
        # A phone is already listening before a short sound starts: pad clips under 1 s with silence
        # in front, as the training pipeline does (features.device_hops).
        pcm16 = np.pad(pcm16, (max(0, WINDOW_SAMPLES - len(pcm16)), 0))
        app = app_scores(pcm16, yamnet, head, pools)
        ref = clip_hop_scores(keras_head, fp)
        n = min(len(app), len(ref))
        diff = float(np.abs(app[:n] - ref[:n]).max()) if n else 0.0
        worst = max(worst, diff)
        a_app, a_ref = alerts(app, thresholds, hits), alerts(ref, thresholds, hits)
        same_alerts += a_app == a_ref
        compared += 1
        if a_app != a_ref:
            print(f"  alerts differ: {Path(fp).name}: app {a_app} vs pipeline {a_ref} (max score diff {diff:.4f})")
    result = {"clips": compared, "max_score_diff": worst, "same_alerts": same_alerts,
              "head_input": head_in, "pools": pools, "run": cfg["_meta"]["run"]}
    print(json.dumps(result, indent=2))
    (selected_run_dir(cfg["_meta"]["run"]) / "app_pipeline_check.json").write_text(json.dumps(result, indent=2))
    return 0 if same_alerts == compared else 1


if __name__ == "__main__":
    sys.exit(main())
