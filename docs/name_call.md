# Beyond the eight classes: name calls and other sounds

SAHARA keeps one reliable, fixed set of eight trained alert classes (`SOUND_CLASSES` in `src/config.py`).
Two layers sit beside it. Neither can block or delay a safety alert.

| Layer | What the user sees | Priority | Model |
|---|---|---|---|
| Trained classes (8) | Full-screen alert, vibration, wristband | High; smoke alarm, siren and glass break fire on one window | YAMNet + trained Dense head |
| Name call | "Raj is calling you" or "Someone is calling you": full-screen alert, its own vibration, wristband id `0x20` | High, but not safety-critical | Pretrained speech embeddings + "Aryan" head; pretrained speaker encoder |
| Other sounds | Quiet notice ("Water tap, faucet", "Cat", "Shout") in a low-importance notification and the history | Low: no vibration, no wristband, silent while a trained class fires | YAMNet's own 521 AudioSet scores |

## Why not just train more classes?

The eight classes are the ones a Deaf user must never miss, so each gets curated data, its own threshold
(`threshold_search.py`) and a cost-weighted loss. Adding a class costs a dataset and a retrain, and every extra class
is another source of false alarms that wears out trust in the real ones. YAMNet already knows 521 AudioSet classes, so
the open-set layer shows those as information without promising safety-grade accuracy. Classes that overlap the
eight (Doorbell, Siren, Knock, Bark...) are reported only through the trained head (`OTHER_SOUND_COVERED_BY_HEAD`).
Constant or uninformative ones (Speech, Silence, room tone, TV, music) are never reported (`OTHER_SOUND_IGNORED`).

## Name calls

Every 0.5 s, when YAMNet hears speech, the phone checks the last 2 s:

1. **Is someone saying "Aryan"?** openWakeWord's `melspectrogram.tflite` and `embedding_model.tflite` (Google's
   speech_embedding, pretrained on many speakers for keyword spotting) give one 96-d embedding per 80 ms. A logistic
   head over 8 consecutive embeddings (1.3 s) scores the name (`src/personalization/name_head.py`,
   `core/NameSpotter.kt`). Trained on the team's 38 "Aryan" clips (plus noisy copies), 19 similar-name clips and the
   Indian background scenes. **This works for anyone**, enrolled or not.
2. **Who is calling?** Only after step 1 fires, the same audio goes through the GE2E speaker encoder (Resemblyzer,
   3-layer LSTM, run in pure Kotlin from `speaker_encoder.bin`, `core/SpeakerEncoder.kt`). Its 256-d voice vector is
   compared by cosine with each enrolled caller's centroid (`core/PrototypeBank.kt`). At or above 0.70 the caller is
   named; below it the alert says "Someone".

Enrolment (Settings → People who call you): the caller holds the phone and says "Aryan" as they normally would. Only
the 256 numbers are stored (`SaharaPrefs.saveCallers`), never the audio. Three recordings per person work best.

Measured on the team recordings (`docs/name_call_eval.md`, leave-one-recorder-out so voices are unseen):

- 82% of calls caught at the equal-error threshold; ROC AUC 0.91 against background scenes, 0.73 against
  similar names (Arya, Aryanshi). TV and hawker speech cause most false calls.
- 79% of calls named correctly with four enrolled callers; at the 0.70 default, 84% of unenrolled callers are shown
  as "Someone" instead of a wrong name.

Caveats: each recorder used their own phone and room, so identity is partly channel identity; enrol on the user's
own phone and re-run the evaluation. The spotter knows one name; another user needs their own clips (synthetic TTS
clips are the usual way to get many voices) and a retrain.

## Tried and rejected: synthetic voices

`scripts/make_tts_name_clips.py` makes 1,781 clips of "Aryan" and near-miss names (Arya, Aryanshi, Ayaan, Rayan...)
with 27 macOS voices, including Indian English and Hindi. `eval_name_call.py --tts-dir ... --tts-use both|pos|neg`
adds them to training only; scoring stays on the team's real recordings (leave one recorder out). Every variant was
worse than real recordings alone, so the shipped head does not use them (measured on the first upload, 19 similar-name clips):

| Training data | AUC vs background | AUC vs similar names | Calls caught at EER |
|---|---:|---:|---:|
| Real recordings only (shipped) | 0.895 | 0.719 | 76% |
| + synthetic, both kinds | 0.835 | 0.615 | 71% |
| + synthetic similar names only | 0.865 | 0.686 | 74% |
| + synthetic "Aryan" only | 0.825 | 0.547 | 76% |

The computer voices differ too much from people calling across a room on a phone microphone; the head partly learns
"synthetic vs real" instead of the name. More **real** recordings of similar names are the way to improve this.

## Real similar names (second upload, 2026-10-30)

The team added 50 real similar-name clips. They have no named recorder, so each is scored in one leave-one-recorder-out
fold and trained on in the others. Same 38 calls and 69 similar names in both rows:

| Training data | AUC vs background | AUC vs similar names | AUC vs the 19 earlier similar names | Calls caught at EER | Similar names raising a call |
|---|---:|---:|---:|---:|---:|
| First upload only (earlier head) | 0.895 | 0.609 | 0.719 | 68% | 61% |
| + 50 real similar names (shipped) | 0.872 | 0.743 | 0.755 | 79% | 36% |

The earlier head was close to chance on the new clips (AUC 0.567). The gain on the 19 earlier clips, which share no
session with the new ones, shows it is not just the new session being learned. Reproduce the first row with
`eval_name_call.py --score-only-unattributed`.

## Rebuild the assets

```bash
pip install -r requirements-personalization.txt   # on macOS, if tflite-runtime fails: pip install --no-deps openwakeword==0.5.1
python scripts/organise_recordings.py --download --source data/interim/drive_recordings --dest data/raw
mkdir -p data/interim/name_wav
for f in data/raw/personalization/aryan/*/*.* data/raw/indian_ambient/background/*.*; do
  case "$f" in *.csv) continue;; esac
  ffmpeg -loglevel error -y -i "$f" -ac 1 -ar 16000 "data/interim/name_wav/$(basename "${f%.*}").wav"
done
python scripts/eval_name_call.py --wav-dir data/interim/name_wav --report docs/name_call_eval.md --export app/assets
python app/tool/export_yamnet_tflite.py          # now also exports YAMNet scores + yamnet_class_map.csv
python app/tool/sync_assets.py --yamnet models/exported/yamnet.tflite
```

The name-call assets (`name_head.json`, `speaker_encoder.bin`, `melspectrogram.tflite`, `embedding_model.tflite`) and
`other_sounds.json` are committed, so name calls work from a fresh checkout; see `app/assets/NOTICE.md` for their
licences. Other sounds also need the re-exported `yamnet.tflite` and its class map. Any missing asset switches its
feature off and the home screen says which.

## Licences

- Resemblyzer GE2E weights: Apache-2.0.
- openWakeWord code: Apache-2.0. Its bundled pretrained model files are CC BY-NC-SA 4.0 (non-commercial), which fits
  an academic capstone. A commercial release would need speech embeddings with a permissive licence.
- YAMNet: Apache-2.0.
