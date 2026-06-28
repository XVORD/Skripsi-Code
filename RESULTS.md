# Final Results

## Global W60, Unseen-Subject Test

Evaluation unit: 794 overlapping W60 windows from 9 videos belonging to 3 held-out participants.

| Metric | Value |
|---|---:|
| Accuracy | 75.44% |
| Macro-F1 | 73.25% |
| Suspicious precision | 67.39% |
| Suspicious recall | 63.92% |
| Suspicious F1 | 65.61% |
| TN / FP / FN / TP | 413 / 90 / 105 / 186 |

## Learned-Stagewise W60, Same Unseen Subjects

| Target | Accuracy | Precision | Recall | F1 | TP / TN / FP / FN |
|---|---:|---:|---:|---:|---:|
| Looking away | 76.95% | 53.55% | 57.07% | 55.26% | 113 / 498 / 98 / 85 |
| Offscreen gaze | 75.44% | 65.08% | 60.52% | 62.72% | 164 / 435 / 88 / 107 |

The larger 5,249-window all-video export is diagnostic only and must not be reported as unseen-subject test performance.

## Decision-Support Integration

| Configuration | Accuracy | Macro-F1 | Precision S. | Recall S. | F1 S. |
|---|---:|---:|---:|---:|---:|
| Global LSTM W60 | 75.44% | 73.25% | 67.39% | 63.92% | 65.61% |
| Global + learned-stagewise evidence | 79.35% | 78.33% | 69.18% | 78.69% | 73.63% |

## Separate Perception Evaluations

| Module and unit | N | F1 |
|---|---:|---:|
| Face present, unique frames | 30,000 | 98.58% |
| Multiple faces, unique frames | 30,000 | 90.51% |
| Phone presence, CITW images | 1,500 | 98.59% |
| Phone localization, CITW boxes | 2,300 | 88.35% |

These values use different evaluation units and are not averaged into an overall system metric.
