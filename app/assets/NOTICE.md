# Third-party model files in this folder

| File | Source | Licence |
|---|---|---|
| `melspectrogram.tflite`, `embedding_model.tflite` | openWakeWord 0.5.1 (David Scripka), re-implementing Google's speech_embedding (TF Hub) | Model files: CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). Code: Apache-2.0 |
| `speaker_encoder.bin` | Resemblyzer 0.1.4 (Resemble AI) GE2E voice encoder weights, repacked by `src/personalization/speaker.py` | Apache-2.0 |
| `yamnet.tflite`, `yamnet_class_map.csv` (when present) | Google YAMNet (TF Hub) | Apache-2.0 |

`name_head.json` is SAHARA's own head, trained on the team's recordings by `scripts/eval_name_call.py --export`.
The non-commercial licence of the openWakeWord files suits this academic project; replace them before any commercial use.
