# Looking-Away Subject-Split Evaluation

- Split manifest: `models/global_lstm_w60_normalized/split_manifest.json`
- Display name: learned temporal model W60, 107 inputs, target rule masked
- Implementation type is preserved in the accompanying JSON for reproducibility.
- Train/validation/test subjects: 14/3/3
- Test unit: 794 W60 windows from 9 held-out-subject videos

| Accuracy | Precision | Recall | F1 positive | Macro-F1 | TP/TN/FP/FN |
|---:|---:|---:|---:|---:|---:|
| 76.95% | 53.55% | 57.07% | 55.26% | 69.87% | 113/498/98/85 |
