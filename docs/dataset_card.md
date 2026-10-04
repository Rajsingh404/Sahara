# SAHARA Dataset Card

**Project**: SAHARA (Sound-based Alert & Haptic Assistance for Real-time Awareness)
**Task**: multi-label detection of 8 safety sounds, plus `background`, for Deaf / Hard-of-Hearing users
**Last updated**: 2026-10-04 (Phase 1 of the build plan, branch `feat/integration`)
**Source of truth for counts**: `data/metadata/coverage.txt`, written by `python src/data/manifest_builder.py`

Class order is fixed by `SOUND_CLASSES` in `src/config.py`:
`smoke_alarm, doorbell, siren, knocking, dog_bark, baby_cry, glass_break, appliance_beep`.
Native-label mappings live only in `src/data/label_map.py`.

---

## Coverage (clips per class × source, train / val / test)

| Class | AudioSet | ESC-50 | FSD50K | Team (Indian) | Synthetic (train only) | **Real total** | **Real test** |
|---|---|---|---|---|---|---|---|
| smoke_alarm | 14/3/3 | – | – | – | 42/0/0 | **20** | **3** ⚠ |
| doorbell | 14/3/3 | – | 100/22/22 | 0/0/16 | 342/0/0 | 180 | 41 |
| siren | 19/8/4 | 35/2/3 | 91/19/22 | – | 435/0/0 | 203 | 29 |
| knocking | 14/0/0 | 26/11/3 | 254/59/60 | 14/0/0 | 924/0/0 | 441 | 63 |
| dog_bark | 6/2/1 | 28/3/9 | 375/82/79 | 0/0/11 | 1227/0/0 | 596 | 100 |
| baby_cry | 16/3/4 | 33/4/3 | – | 6/0/0 | 165/0/0 | **69** | **7** ⚠ |
| glass_break | 54/10/11 | 30/5/5 | 355/78/77 | – | 1317/0/0 | 625 | 93 |
| appliance_beep | 12/8/3 | 33/1/6 | 145/30/31 | 9/0/7 | 597/0/0 | 285 | 47 |
| background | – | – | – | 37/0/23 | – | 665 (DESED 605) | 115 |

Targets from the build plan: ≥ 150 real clips per class and ≥ 20 real clips in test.
**smoke_alarm and baby_cry fall short.** Every other class meets both.

Manifest rows: 8,133 (train 7,192, val 443, test 498); 8,129 unique audio files
(a few FSD50K clips carry two target labels and so have two rows).
Real rows: FSD50K 1,901, DESED 605, ESC-50 240, AudioSet 215, team recordings 123. Synthetic rows: 5,049.

---

## Sources

| Source | Licence | Location | Role |
|---|---|---|---|
| FSD50K (Fonseca et al. 2021) | per clip: CC0 / CC BY / CC BY-NC | Google Drive for Desktop, `SAHARA_DRIVE_DATASETS/FSD50K` (streamed) | targets |
| AudioSet (Gemmeke et al. 2017) | labels CC BY 4.0; audio from YouTube | `data/raw/audioset/<class>/` | targets |
| ESC-50 (Piczak 2015) | CC BY-NC 3.0 (academic use) | `data/raw/esc50/` via `src/data/download_esc50.py` | targets |
| DESED real soundscapes (Turpault et al. 2019) | per clip, see `desed/real/license_public_eval.tsv` | `data/raw/desed/real/` | background |
| Team recordings (Advika, Mahi, Prachi, Raj; Sept 2026) | owned by the team | `data/raw/indian_ambient/`, built by `scripts/organise_recordings.py` | Indian-context targets + background |
| Synthetic mixes | derived from the above | `data/raw/synthetic_mixed/` (local, 1.1 GB) | train-only augmentation |
| iNoise (IEEE DataPort) | IEEE DataPort open access | **not present**: needs a manual, logged-in download | background (planned) |

Audio is never committed. `data/raw/` is git-ignored; only the manifests and `coverage.txt` are.

---

## Label-mapping fixes (Phase 1)

Checked against FSD50K `vocabulary.csv` and AudioSet `class_labels_indices.csv`.

**FSD50K**
- There is no smoke-alarm, baby-cry or `Beep_buzzer` label in FSD50K. The old map named
  `Smoke_detector_smoke_alarm`, `Baby_cry_infant_cry` and `Beep_buzzer`, which match nothing.
  (`Crying_and_sobbing` exists but is adult crying, so it is not used.)
- `glass_break`: `Glass` → `Shatter`. Of 1,241 `Glass` clips, 731 are clinking only; the 510 tagged `Shatter` are kept.
- `appliance_beep`: generic `Alarm` removed. Its 1,864 clips are mostly telephones (685), vehicle/car alarms,
  ringtones, doorbells and sirens. Only `Microwave_oven` (206) remains. No FSD50K clip carries a smoke-alarm tag,
  so `Alarm` cannot be routed to `smoke_alarm` either.

**AudioSet**: four hard-coded ids were wrong (the build plan repeated one of them).

| Class | Old id | Actually | Now |
|---|---|---|---|
| glass_break | `/m/07rwj3x` | Whoop | `/m/07rn7sz` Shatter |
| knocking | `/m/07qnq_y` | Thump, thud | `/m/07r4wb8` Knock |
| appliance_beep | `/m/03cl9h` | Ice cream truck | `/m/02fs_r` Beep, bleep; `/m/0fx9l` Microwave oven; `/m/046dlr` Alarm clock |
| appliance_beep | `/m/029bxz` | (no such id) | – |

A display-name fallback had also added `Glass`, `Breaking` and `Dog`. It is gone; only exact ids are used, and
`dog_bark` is `Bark` only. Of the 188 AudioSet clips from the earlier run, **64 did not carry their class's label**
(9 "appliance_beep" ice-cream trucks, 21 "glass_break" whoops/music, 15 "knocking" thumps, 19 generic dog sounds).
They were moved to `data/interim/audioset_mislabelled/`. The AudioSet scanner now looks up every clip's real labels in
the segment CSVs instead of trusting the folder it was downloaded into.

**DESED**: every DESED clip used to be `background`, including clips with `Dog` (barking) and `Alarm_bell_ringing`.
Those 224 clips and the 76 unlabelled in-domain clips are now dropped (`DESED_EXCLUDED_EVENTS` in `label_map.py`).

### Coverage before → after

| Class | Before (old manifest) | After |
|---|---|---|
| smoke_alarm | 20 | 20 |
| doorbell | 164 | 180 |
| siren | 152 | 203 |
| knocking | 402 | 441 |
| dog_bark | 556 | 596 |
| baby_cry | 20 | 69 |
| glass_break | 1,275 | 625 |
| appliance_beep | 2,093 | 285 |
| background | 905 | 665 |

The large drops (glass_break, appliance_beep, background) are wrong labels removed, not data lost.

---

## Splits

70 / 15 / 15 by **recording group**, stratified by each group's rarest label (`split_rows()` in `manifest_builder.py`).

- **Group id = upstream source.** AudioSet and DESED both cut clips from YouTube, so both use `yt:<video id>`.
  FSD50K and ESC-50 both come from Freesound, so both use `fs:<freesound id>`. The same video or Freesound clip
  can therefore never sit in train in one dataset and test in another. All rows of a group go to one split,
  including multi-label clips (the old split was per label and could put one clip in train and test).
- **Team recordings keep the split written by `organise_recordings.py`**: per label, ~60 / 40 train / test across
  recording conditions; no validation rows. They are tagged `source_type=self_recorded`.
- **Synthetic mixes are train-only** and grouped with the real clip they were made from. A mix whose source is not
  in train is dropped. Validation and test are real audio only.
- `tests/test_dataset_split.py` checks all of the above.

---

## Synthetic mixes

`python src/preprocessing/synthetic_mixing.py`: each of the 1,683 real train-split target clips is mixed onto 3
random train-split background beds (DESED 423, team background 37) at an SNR drawn uniformly from −5 to +10 dB,
then peak-limited to 0.99. 5,049 mixes. `synthetic_metadata.csv` records the source clip, its recording id, the bed
and the SNR. Only train-split beds are used, so no validation/test audio reaches training. iNoise beds will be added
when iNoise is downloaded. Synthetic rows are always `source_type=synthetic_mixed` and are never reported as
real-world performance.

---

## Team recordings (Indian context)

180 clips recorded on phones in September 2026. `data/recordings/label_map.csv` (Drive file ids) is the source of
truth; `scripts/organise_recordings.py --source <Drive Indian_Ambient folder> --dest data/raw` rebuilds the layout.

Split **per label** (not per person; every recording has its own conditions): ~40 % of each label goes to test,
spread across the label's recording conditions (Drive folder × recorder); identical clips stay together.
Details: `data/recordings/README.md`.

| Label | Train | Test |
|---|---|---|
| appliance_beep | 10 | 6 |
| baby_cry | 4 | 2 |
| knocking | 8 | 6 |
| dog_bark | 7 | 4 |
| doorbell | 10 | 6 |
| background | 36 | 24 |
| smoke_alarm, siren, glass_break | 0 | 0 |
| "Aryan" positive / negative (personalization) | 23 / 11 | 15 / 8 |

Used as an Indian-context fine-tuning set (train side) and evaluation set (test side), not as the main training set.
Every recorded class is now on both sides. smoke_alarm, siren and glass_break have no team recordings yet.

---

## Known gaps

1. **smoke_alarm (20 real clips, 3 in test) and baby_cry (69, 7 in test)** are below the ≥ 150 / ≥ 20 targets.
   The AudioSet top-up was stopped by YouTube's bot check ("Sign in to confirm you're not a bot") after 4 parallel
   downloads; see `data/raw/audioset/download_manifest.csv`. Neither class is in FSD50K, and ESC-50 has no smoke alarm.
2. **iNoise is missing** everywhere (Mac, Drive, old zip). It needs a free IEEE DataPort login.
3. **No team recordings of smoke_alarm, siren or glass_break.**
4. FSD50K is streamed from Google Drive for Desktop; the first read of each file is slow (~2 s).
