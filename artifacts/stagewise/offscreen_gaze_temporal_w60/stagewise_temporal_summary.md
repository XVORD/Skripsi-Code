# Offscreen-Gaze Subject-Split Evaluation

- Split manifest: `models/global_lstm_w60_normalized/split_manifest.json`
- Display name: learned temporal model W60 with temporal statistics, 107 inputs, target rule masked
- Train/validation/test subjects: 14/3/3
- Test unit: 794 W60 windows from 9 held-out-subject videos

| Accuracy | Precision | Recall | F1 positive | Macro-F1 | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|
| 75.44% | 65.08% | 60.52% | 62.72% | 72.20% | 164/435/88/107 |
