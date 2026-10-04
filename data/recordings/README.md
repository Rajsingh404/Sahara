# Team recordings

Audio recorded by the four team members (Advika, Mahi, Prachi, Raj) in September 2026, from the shared Google Drive
folder (one sub-folder per person). Audio never goes in git; this folder holds only the labels and generated metadata.

| File | What it is |
|---|---|
| `label_map.csv` | Source of truth. One row per Drive file: Drive id, original path, recorder, label, new filename, duration, device, notes. Edit this to fix a label, then re-run the script. |
| `metadata/` | The `metadata.csv` files `scripts/organise_recordings.py` writes next to the audio, committed for review. |

## Rebuild locally

```bash
python scripts/organise_recordings.py --download --source data/interim/drive_recordings --dest data/raw
```

This downloads every clip by Drive id and writes:

```
data/raw/indian_ambient/
├── metadata.csv            schema from docs/dataset_card.md + recorder, split, duration_sec, source_path
├── target_classes/         <class>__<recorder>__NN.m4a|mp3
└── background/             background_<scene>__<recorder>__NN.m4a|mp3
data/raw/personalization/aryan/
├── positive/               people saying "Aryan"        + metadata.csv
├── negative/               other / similar names         + metadata.csv
└── unverified/             not yet sorted (see below)    + metadata.csv
```

`manifest_builder.py` picks up `data/raw/indian_ambient/` with no code changes. A copy of the organised audio lives in
the project's shared folder at `sahara-data/`.

## Split by recorder

Each clip's `split` comes from who recorded it, so one person's voice, phone and room never sit on both sides.
Default test recorders: **Mahi, Prachi** (`--test-recorders` changes it). Because each person recorded different
sounds, no recorder split covers every class on both sides:

| Label | Train (Advika, Raj) | Test (Mahi, Prachi) |
|---|---|---|
| appliance_beep | 9 (microwave) | 7 (washing machine) |
| baby_cry | 6 | 0 |
| knocking | 14 | 0 |
| dog_bark | 0 | 11 |
| doorbell | 0 | 16 |
| background | 37 | 23 |
| aryan (positive) | 19 | 4 |
| not_aryan (negative) | 13 | 6 |
| unverified | 0 | 15 |

These clips are best used as an Indian-context evaluation and fine-tuning set on top of FSD50K/AudioSet, not as a
standalone training set.

## Open labelling questions

- **Prachi/Aryan (15 clips)**: all named "Copy of New Recording N", so positive vs negative can't be told from the
  name. Ten were added 2026-09-18 and five more on 2026-10-04. Kept in `unverified/` until someone listens.
- **Mahi/Aryan "name", "name 2", "name 3"**: filed as negatives (assumed another name); needs a listen.
- **Advika "door bang" (4 clips)**: labelled `knocking`; could be door slams.
- **Advika door open/close and drawer bang (7 clips)**: moved to `background` as hard negatives, not knocks.
- **Mahi washing machine**: the m4a clips are labelled `appliance_beep` on the assumption they contain the end-of-cycle
  beep; `washing-spinning.mp3` is `background`.
- **mp3s with no recording metadata** (Advika baby cry 5–6 and microwave 7–9; Mahi dog 4, 7, 8, doorbell 1–2,
  washing.mp3): confirm they were recorded by the team and not downloaded. `microwave 9.mp3` is 126 s long.
- **Mahi dog barking**: several are exports of older phone videos (2022, 2024, earlier in 2026), and `dog5` is a screen
  recording.
