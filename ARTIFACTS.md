# Artifact Provenance

| Path | Purpose | Evaluation boundary |
|---|---|---|
| `models/global_lstm_w60_normalized/` | Selected global checkpoint and training history | Subject split, W60 |
| `artifacts/global/global_metrics.json` | Final global and decision-support metrics | 794 unseen-subject windows |
| `artifacts/stagewise/` | Learned looking-away and offscreen-gaze reports | Same subject manifest; 794 test windows |
| `artifacts/face_modules/metrics.json` | Face presence and multiple-face metrics | 30,000-frame face-presence test and separate 940-frame multiple-face supplement |
| `artifacts/object_detection/citw_metrics.json` | CITW summary reported in the thesis/paper | External-device evaluation summary |
| `artifacts/figures/` | Figures referenced by the IEEE paper | Derived from the listed artifacts |

## Important Boundary

The CITW aggregate reported in the final thesis was executed independently on another device. This repository stores the reported confusion counts and a representative contact sheet, but not the complete external image set or raw prediction dump. It should therefore be treated as a reported external experiment, not as a locally reproduced result.

Participant videos and complete feature caches are private research data and are intentionally excluded from the public package.
