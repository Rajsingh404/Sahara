# SAHARA Baseline Evaluation

Metrics use per-class one-vs-rest decisions at a 0.5 threshold. mAP uses the prediction scores and is less threshold-sensitive.

| Class | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| smoke_alarm | 0.500 | 1.000 | 0.667 | 0.917 |
| doorbell | 1.000 | 0.143 | 0.250 | 0.663 |
| siren | 0.600 | 1.000 | 0.750 | 1.000 |
| knocking | 1.000 | 0.600 | 0.750 | 0.863 |
| dog_bark | 1.000 | 0.333 | 0.500 | 0.806 |
| baby_cry | 1.000 | 0.333 | 0.500 | 1.000 |
| glass_break | 0.429 | 0.429 | 0.429 | 0.662 |
| appliance_beep | 0.000 | 0.000 | 0.000 | 0.354 |
| **Macro average** | **0.691** | **0.480** | **0.481** | **0.783** |

## Error analysis

The baseline remains sensitive to class imbalance. The small smoke-alarm and baby-cry subsets and acoustic overlap among alarm/beep-like events make thresholded recall unstable. Indian-context recordings and threshold calibration are deliberately deferred to the next phase.
