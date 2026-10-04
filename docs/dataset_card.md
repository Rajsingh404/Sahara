# SAHARA Dataset Card

**Project**: SAHARA — Safety-Aware Hearing Assistance with Real-time Alerts  
**Version**: Phase 2-3 (baseline pipeline complete; cached FSD50K embeddings available)  
**Last updated**: 2026-08-28  
**Task**: Multi-label 8-class safety-critical sound detection for Deaf/Hard-of-Hearing users  
**Model architecture**: Frozen YAMNet backbone + trainable dense classifier head. TFLite export is a next-phase task.

---

## Target Classes

| Canonical Label | Safety Scenario |
|---|---|
| `smoke_alarm` | Fire/smoke alert |
| `doorbell` | Visitor / delivery notification |
| `siren` | Emergency vehicle, building alarm |
| `knocking` | Door knock |
| `dog_bark` | Pet/visitor alert |
| `baby_cry` | Infant distress |
| `glass_break` | Intrusion / accident |
| `appliance_beep` | Microwave, washing machine, oven timer |

**Background class** (`background`): Non-target ambient audio used as negative examples. Sourced from DESED real soundscapes and (Phase 4) Indian ambient recordings.

---

## Data Sources

### FSD50K

- **Source**: [Fonseca et al., 2021](https://arxiv.org/abs/2010.00475) — Freesound Dataset 50K
- **License**: Varies per clip (Creative Commons); see FSD50K.metadata for per-file licenses
- **Location**: raw audio is not retained locally after caching; reproducible source download is implemented in `src/data/download_fsd50k.py`.
- **Status**: Ground-truth-derived manifest rows and cached YAMNet embeddings are available locally. Re-download the source audio only if rebuilding the embedding cache from scratch.
- **Manifest rows matched**: 4,494 clips across 8 target classes (from ground-truth CSVs)
- **Embeddings extracted**: 4,494 target-class FSD50K clips

### AudioSet

- **Source**: [Gemmeke et al., 2017](https://research.google.com/audioset/) — Google AudioSet
- **License**: CC BY 4.0 (metadata); individual YouTube clips subject to YouTube ToS
- **Location**: `data/raw/audioset/<class_name>/` — **locally available on disk**
- **Download method**: yt-dlp, 10-second segments at 16 kHz mono WAV
- **Status**: 161 clips downloaded (20–21 per class), all locally available and embedded
- **Failed downloads log**: `data/raw/audioset/failed_downloads.csv`

### DESED

- **Source**: [Turpault et al., 2019](https://dcase.community/challenge2019/task4-sound-event-detection-in-domestic-environments) — Domestic Environment Sound Event Detection
- **License**: See individual clip licenses in DESED metadata
- **Location**: `data/raw/desed/real/` — locally available on disk
- **Status**: ~884–904 clips (background negatives); some upstream URLs fail to resolve — logged and skipped
- **Role**: Background/negative soundscape audio in manifests; not used for training embeddings in current baseline (target-class-only training)

### iNoise (Indian Noise Database)

- **Source**: [Kopparapu, Sheikh, Thanneeru, IEEE DataPort](https://ieee-dataport.org/documents/inoise-indian-noise-database) (DOI: 10.21227/w3xm-jn45)
- **License**: IEEE DataPort Open Access
- **Location**: `$SAHARA_DRIVE_DATASETS/iNoise/` (raw) and `$SAHARA_DRIVE_DATASETS/iNoise_chunks/` (5s chunks) — **Drive-resident**
- **Categories**: Outdoor (Autorickshaw, Bus, Highway, Railway Station, Street) and Indoor (Airport, Cafeteria, Home, Train, Workplace)
- **Role**: Background/negative ambient noise (mapped to `background` label, `source_type: real`)

### Synthetic Mixed Audio (`synthetic_mixed`)

- **Source**: Additive mixing pipeline (`src/preprocessing/synthetic_mixing.py`) combining target clips (FSD50K/AudioSet) with background chunks (iNoise/DESED) at randomized SNR (-5 dB to 10 dB).
- **Location**: `$SAHARA_DRIVE_DATASETS/SyntheticMixed/` with companion `$SAHARA_DRIVE_DATASETS/SyntheticMixed/synthetic_metadata.csv` — **Drive-resident**
- **Role**: Data augmentation and background robustness tuning for the 8 target classes.
- **Academic Honesty Notice**: Every synthetic clip carries `source_type: synthetic_mixed` in manifests to ensure real and synthetic data are never silently conflated.
- **Important Distinction**: `synthetic_mixed` and `iNoise` data do **NOT** substitute for the still-pending Indian ambient field recordings (target-class-in-Indian-context) or the personalization dev-set — both remain required, separate next steps.

### Indian Ambient (Phase 4 — first batch recorded)

- **Status**: 123 clips recorded by the team in September 2026 (63 target-class, 60 background), organised by `scripts/organise_recordings.py` from `data/recordings/label_map.csv`. See `data/recordings/README.md` for counts, the recorder split and open labelling questions.
- **Classes covered**: doorbell 16, appliance_beep 16, knocking 14, dog_bark 11, baby_cry 6, background 60 (kitchen, temple, hawker, rain, TV, door/drawer, washing-machine spin). No smoke_alarm, siren or glass_break yet.
- **Split**: by recorder (default test = Mahi, Prachi), via the extra `recorder` and `split` columns in `metadata.csv`. `manifest_builder.py` does not read `split` yet; it groups by `timestamp` (date + recorder).
- **Location**: `data/raw/indian_ambient/`
- **Hot-swap**: Re-running `src/data/manifest_builder.py` with NO code changes automatically incorporates this data once present. **Validated 2026-08-28 — test passed.**
- **Expected structure**:
  ```
  data/raw/indian_ambient/
  ├── metadata.csv          ← Required: schema below
  ├── target_classes/       ← Safety-critical sound recordings in Indian contexts
  └── background/           ← Ambient negatives: traffic, temple bells, monsoon rain, etc.
  ```
- **`metadata.csv` schema**:

  | Column | Type | Description |
  |---|---|---|
  | `filename` | str | Filename only (not full path) |
  | `class` | str | One of the 8 canonical class names, or `background` |
  | `is_background` | bool | `true` for files in `background/`, `false` for `target_classes/` |
  | `location_type` | str | e.g. `urban_street`, `apartment`, `market` |
  | `recording_device` | str | e.g. `OnePlus_Nord_CE4`, `Zoom_H1n` |
  | `distance_m` | float | Approximate distance in metres (source to mic) |
  | `timestamp` | str | ISO-8601 or YYYY-MM-DD session tag (used as recording_id for split stability) |
  | `notes` | str | Free text — acoustic conditions, unusual interference, etc. |

---

## Per-Class Coverage Table

*Generated by `src/data/manifest_builder.py` on 2026-08-28. Manifest includes all sources present under `data/raw/`.*

| Class | audioset | desed | fsd50k | **total** | Note |
|---|---|---|---|---|---|
| `smoke_alarm` | 20 | 0 | 0 | **20** | ⚠ LOW — FSD50K has no `Smoke_detector_smoke_alarm` label in dev/eval; AudioSet only |
| `doorbell` | 20 | 0 | 144 | **164** | FSD50K pending sync |
| `siren` | 20 | 0 | 132 | **152** | FSD50K pending sync |
| `knocking` | 20 | 0 | 373 | **393** | FSD50K pending sync |
| `dog_bark` | 20 | 0 | 536 | **556** | FSD50K pending sync |
| `baby_cry` | 20 | 0 | 0 | **20** | ⚠ LOW — FSD50K has no `Baby_cry_infant_cry` label in dev/eval; AudioSet only |
| `glass_break` | 21 | 0 | 1241 | **1262** | FSD50K pending sync |
| `appliance_beep` | 20 | 0 | 2068 | **2088** | FSD50K pending sync |
| `background` | 0 | 905 | 0 | **905** | DESED real soundscapes |
| **Total target** | **161** | — | **4494** | **4655** | 161 embedded; 4494 blocked on Drive sync |

**Manifest totals**: full=5,560 · train=3,892 · val=834 · test=834  
**Embeddings extracted**: 4,655 target clips (4,494 FSD50K + 161 AudioSet) · train=3,259 · val=698 · test=698. Background rows are retained in the manifest but excluded from this 8-class baseline.

---

## Manifest Structure

All data is unified into `data/metadata/`:

| File | Rows (header+1) | Description |
|---|---|---|
| `full_manifest.csv` | 5,560 | All clips from all sources (`filepath, label, source_dataset, duration_sec, recording_id`) |
| `train_manifest.csv` | 3,892 | 70% stratified by class + recording session |
| `val_manifest.csv` | 834 | 15% — held-out for early stopping |
| `test_manifest.csv` | 834 | 15% — held-out for final evaluation |

**Split strategy**: Stratified by class label, split at recording-session level (`recording_id`) to prevent data leakage from the same session across train/val/test.

---

## Baseline Training Results (2026-08-28)

**Training data**: 3,259 target clips (FSD50K + AudioSet)  
**Validation data**: 698 clips · **Test data**: 698 clips  
**Augmentation**: Disabled for this reproducible baseline. The raw-waveform augmenter is available through `build_features_cache.py --augment` for future training experiments.

### Training

| Metric | Train | Val |
|---|---|---|
| Loss / accuracy (final epoch) | 0.3261 / 0.7969 | 0.2263 / 0.8095 |

Optimizer: Adam lr=1e-3 · Loss: cost-sensitive weighted binary crossentropy · Architecture: Dense(256)→Drop(0.3)→Dense(128)→Drop(0.3)→Dense(8, sigmoid). Early stopping completed after 9 epochs; the best checkpoint was re-evaluated on 2026-08-29.

### Evaluation (test set, 698 clips, threshold=0.5)

| Class | Precision | Recall | F1 | AP |
|---|---|---|---|---|
| `smoke_alarm` | 0.000 | 0.000 | 0.000 | 0.090 |
| `doorbell` | 0.682 | 0.395 | 0.500 | 0.632 |
| `siren` | 0.333 | 0.333 | 0.333 | 0.250 |
| `knocking` | 0.826 | 0.655 | 0.731 | 0.751 |
| `dog_bark` | 1.000 | 0.893 | 0.943 | 0.956 |
| `baby_cry` | 0.000 | 0.000 | 0.000 | 0.669 |
| `glass_break` | 0.881 | 0.895 | 0.888 | 0.937 |
| `appliance_beep` | 0.860 | 0.790 | 0.824 | 0.923 |
| **Macro avg** | **0.573** | **0.495** | **0.527** | **mAP = 0.651** |

> Note: This is a real but imbalanced baseline. Smoke alarm and baby cry each have only 20 clips; both have zero thresholded recall despite the safety weighting. Do not use these numbers as final deployment performance; tune thresholds and expand underrepresented classes in the next phase.

**Confusion matrix**: `docs/confusion_matrix.png`

---

## Known Gaps

### 1. Raw FSD50K Audio Is Not Retained Locally
- **Status**: The 4,494 FSD50K target clips were embedded and the raw files were subsequently removed to conserve disk space.
- **Impact**: Current training/evaluation artifacts are reproducible from cached embeddings, but rebuilding them from raw audio requires re-running the FSD50K downloader.

### 2. smoke_alarm and baby_cry — Not in FSD50K
- **Status**: Confirmed: `Smoke_detector_smoke_alarm` and `Baby_cry_infant_cry` do not appear in FSD50K dev or eval CSVs. These classes have only 20 clips each (AudioSet).
- **Action**: Pull additional clips from AudioSet unbalanced set via `python src/data/download.py --dataset audioset` (no `--limit-per-class`) and add field recordings.

### 3. Indian Ambient Dataset — First Batch Only
- **Status**: 123 clips recorded (see above). smoke_alarm, siren and glass_break have no Indian recordings yet, and baby_cry/knocking come from one recorder each.
- **Hot-swap design validated**: 2026-08-28 — dummy files added to `data/raw/indian_ambient/` and `manifest_builder.py` re-run with zero code changes; 2 indian_ambient rows appeared correctly. Empty-folder run produced 0 rows gracefully.
- **Novel contribution**: First safety-sound dataset evaluated specifically in Indian urban/domestic acoustic environments.

### 4. Personalization Module — Dev-Set Recorded, Model Not Started
- **Status**: Phase 12. Name-call dev-set for "Aryan" recorded by all four team members: 23 positive, 19 negative (other or similar names such as "Arya", "Aryanshi"), 15 not yet sorted. Layout: `data/raw/personalization/aryan/{positive,negative,unverified}/` with a `metadata.csv` each.

---

## Preprocessing Pipeline

```
Audio file (any format, any SR)
    ↓ load_audio(): librosa.load(sr=16000, mono=True) + peak normalize
    ↓ Drive-stub check (xattr): skip if not locally available
    ↓ YAMNet (TF-Hub https://tfhub.dev/google/yamnet/1)
        → frame-level 1024-d embeddings
        → mean pool → single 1024-d vector per clip
        → cache to data/processed/embedding_cache/<sha256>.npy
    → stack into data/processed/embeddings_{full,train,val,test}.npz
```

**Log-mel spectrograms** (64 mel bins, n_fft=1024, hop=160, sr=16000) computed via `src/preprocessing/features.extract_logmel()` for visualization and report.

**Augmentation** (config-toggleable, `--augment` flag): time-stretch ±10%, pitch-shift ±2 semitones, gain ±6 dB, background noise mix. Embedding-space Gaussian jitter available as fast alternative via `augment_embedding_batch()`.

---

## Roadmap Context

| Phase | Description | Status |
|---|---|---|
| 0–1 | Repo scaffolding, config, system architecture | ✅ Done |
| 2–3 | Dataset acquisition, preprocessing, baseline model | ✅ Pipeline complete; FSD50K sync pending |
| 4 | Indian ambient field recording + personalization dev-set | ⏳ Blocked (not yet recorded) |
| 5–7 | Hyperparameter tuning, cost-sensitive threshold search, FAR evaluation | ⏳ Pending |
| 8–9 | TFLite export, INT8 quantization, on-device inference pipeline | ⏳ Pending |
| 10 | Native Kotlin Android app (Foreground Service + AudioRecord + BLE GATT client) | ⏳ Pending |
| 11 | ESP32 custom wristband firmware (BLE GATT server, haptics, <₹1000 BOM) | ⏳ Pending |
| 12 | Personalization (few-shot prototypical network fused with main classifier) | ⏳ Pending |
| 13–14 | Testing strategy, deployment, GitHub polish, weekly milestone tracking | ⏳ Pending |
