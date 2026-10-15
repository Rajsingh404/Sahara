"""Write the mean training-background window vector for on-device custom-sound matching.

models/exported/personal_center.bin: 1,024 little-endian float32 values, read by the app as
assets/personal_center.bin (CustomSoundMatcher.parseCenter) and copied by app/tool/sync_assets.py.

    python src/personalization/export_center.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.personalization.prototypes import background_center


def main() -> Path:
    center = background_center(np.load(PROCESSED_DATA_DIR / "train_embeddings.npy", mmap_mode="r"),
                               np.load(PROCESSED_DATA_DIR / "train_labels.npy"))[:1024]
    target = MODELS_DIR / "exported" / "personal_center.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(center.astype("<f4").tobytes())
    print(f"{target} ({target.stat().st_size} bytes, |center| = {np.linalg.norm(center):.3f})")
    return target


if __name__ == "__main__":
    main()
