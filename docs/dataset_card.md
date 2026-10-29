# SAHARA Dataset Card

**Project**: SAHARA (Sound-based Alert & Haptic Assistance for Real-time Awareness)
**Task**: multi-label detection of 8 safety sounds, plus `background`, for Deaf / Hard-of-Hearing users
**Last updated**: 2026-10-15 (final dataset build used by run `20261005-195620_b_synthetic`)
**Source of truth for counts**: `data/metadata/coverage.txt`, written by `python src/data/manifest_builder.py`

Class order is fixed by `SOUND_CLASSES` in `src/config.py`:
`smoke_alarm, doorbell, siren, knocking, dog_bark, baby_cry, glass_break, appliance_beep`.
Native-label mappings live only in `src/data/label_map.py`.

---

## Coverage (clips per class × source, train / val / test)

| Class | AudioSet | ESC-50 | FSD50K | Team (Indian) | Synthetic (train only) | **Real total** | **Real test** |
|---|---|---|---|---|---|---|---|
| smoke_alarm | 129/27/27 | – | – | – | 387/0/0 | 183 | 27 |
| doorbell | 67/11/14 | – | 99/24/21 | 17/0/10 | 528/0/0 | 263 | 45 |
| siren | 24/4/6 | 25/4/11 | 95/20/17 | – | 432/0/0 | 206 | 34 |
| knocking | 73/10/14 | 30/8/2 | 256/60/57 | 16/0/11 | 1101/0/0 | 537 | 84 |
| dog_bark | 10/0/0 | 24/5/11 | 376/81/79 | 7/0/4 | 1251/0/0 | 597 | 94 |
| baby_cry | 74/15/17 | 25/7/8 | – | 9/0/5 | 309/0/0 | 160 | 30 |
| glass_break | 71/13/13 | 30/5/5 | 352/79/79 | – | 1359/0/0 | 647 | 97 |
| appliance_beep | 177/37/47 | 30/5/5 | 147/33/26 | 18/0/11 | 1092/0/0 | 536 | 89 |
| background | – | 1228/265/267 | 7008/1494/1498 | 36/0/24 | – | 12,425 (+ DESED 420/98/87) | 1,876 |

Targets from the build plan: ≥ 150 real clips per class and ≥ 20 real clips in test. **Every class meets both.**
(smoke_alarm and baby_cry were raised from 20 clips each with the Hugging Face AudioSet mirror; doorbell, knocking and
appliance_beep from AudioSet's balanced/eval sets.)

Manifest rows: 22,013 (train 17,332, val 2,305, test 2,376); 21,985 unique audio files (a few clips carry two labels).
Real rows: FSD50K 11,901 (of which 10,000 background), ESC-50 2,000 (1,760 background), AudioSet 880, DESED 605,
team recordings 168. Synthetic rows: 6,459 (built before the team's second upload, so its 45 clips are not in any mix).

---

## Sources

| Source | Licence | Location | Role |
|---|---|---|---|
| FSD50K (Fonseca et al. 2021) | per clip: CC0 / CC BY / CC BY-NC | Google Drive for Desktop, `SAHARA_DRIVE_DATASETS/FSD50K` (streamed) | targets |
| AudioSet (Gemmeke et al. 2017) | labels CC BY 4.0; audio from YouTube | `data/raw/audioset/<class>/`: YouTube (yt-dlp) and the Hugging Face mirror `agkphysics/AudioSet` (`download_audioset_hf.py`) | targets |
| ESC-50 (Piczak 2015) | CC BY-NC 3.0 (academic use) | `data/raw/esc50/` via `src/data/download_esc50.py` | 6 categories as targets, the other 44 as background |
| FSD50K non-target sample | as FSD50K | Google Drive | 10,000 seeded background clips without any target-like label (`FSD50K_BACKGROUND_EXCLUDE`) |
| DESED real soundscapes (Turpault et al. 2019) | per clip, see `desed/real/license_public_eval.tsv` | `data/raw/desed/real/` | background |
| Team recordings (Advika, Mahi, Prachi, Raj; Sept 2026) | owned by the team | `data/raw/indian_ambient/`, built by `scripts/organise_recordings.py` | Indian-context targets + background |
| Synthetic mixes | derived from the above | `data/raw/synthetic_mixed/` (local, 1.5 GB) and `synthetic_mixed_test/` (115 MB) | train-only augmentation; separate synthetic test set |
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

`python src/preprocessing/synthetic_mixing.py`: each of the 2,153 real train-split target clips is mixed onto 3 random
train-split background beds (8,692 beds: FSD50K 7,008, ESC-50 1,228, DESED 420, team 36) at an SNR drawn uniformly
from −5 to +10 dB, then peak-limited to 0.99: 6,459 mixes. `synthetic_metadata.csv` records the source clip, its
recording id, the bed and the SNR. Only train-split beds are used, so no validation/test audio reaches training.
A separate synthetic **test** set (`--split test`, 420 mixes of test clips on test beds) is never used for training and
is reported only as synthetic. Synthetic rows are always `source_type=synthetic_mixed` and are never reported as
real-world performance.

---

## Team recordings (Indian context)

275 clips recorded on phones: 180 in September 2026 and 95 in a second upload on 2026-10-30 (recorder not given, filed as `team`). `data/recordings/label_map.csv` (Drive file ids) is the source of
truth; `scripts/organise_recordings.py --source <Drive Indian_Ambient folder> --dest data/raw` rebuilds the layout.

Split **per label** (not per person; every recording has its own conditions): ~40 % of each label goes to test,
spread across the label's recording conditions (Drive folder × recorder); identical clips stay together. Each upload
is split on its own (`upload` column), so new clips never move an earlier one.
Details: `data/recordings/README.md`.

| Label | Train | Test |
|---|---|---|
| appliance_beep | 18 | 11 |
| baby_cry | 9 | 5 |
| knocking | 16 | 11 |
| dog_bark | 7 | 4 |
| doorbell | 17 | 10 |
| background | 36 | 24 |
| smoke_alarm, siren, glass_break | 0 | 0 |
| "Aryan" positive / negative (personalization) | 23 / 41 | 15 / 28 |

Used as an Indian-context fine-tuning set (train side) and evaluation set (test side), not as the main training set.
Every recorded class is now on both sides. smoke_alarm, siren and glass_break have no team recordings yet.

---

## Known gaps

1. **iNoise is missing** everywhere (Mac, Drive, old zip). It needs a free IEEE DataPort login.
2. **No team recordings of smoke_alarm, siren or glass_break**; the Indian-context results cover five classes.
3. The team set is small (168 clips); Indian results use 5-fold cross-validation over all of it (see the report).
4. FSD50K is streamed from Google Drive for Desktop; the first read of each file is slow (~2 s), and a network drop
   can stall a read (the pipeline skips reads that take over 2 minutes).
