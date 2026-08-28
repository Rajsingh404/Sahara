# SAHARA Baseline Evaluation

Metrics use per-class one-vs-rest decisions at a 0.5 threshold. mAP uses the prediction scores and is less threshold-sensitive.

| Class | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|
| smoke_alarm | 0.000 | 0.000 | 0.000 | 0.090 |
| doorbell | 0.682 | 0.395 | 0.500 | 0.632 |
| siren | 0.333 | 0.333 | 0.333 | 0.250 |
| knocking | 0.826 | 0.655 | 0.731 | 0.751 |
| dog_bark | 1.000 | 0.893 | 0.943 | 0.956 |
| baby_cry | 0.000 | 0.000 | 0.000 | 0.669 |
| glass_break | 0.881 | 0.895 | 0.888 | 0.937 |
| appliance_beep | 0.860 | 0.790 | 0.824 | 0.923 |
| **Macro average** | **0.573** | **0.495** | **0.527** | **0.651** |

## Error analysis

The baseline remains sensitive to class imbalance. The small smoke-alarm and baby-cry subsets and acoustic overlap among alarm/beep-like events make thresholded recall unstable. Indian-context recordings and threshold calibration are deliberately deferred to the next phase.
