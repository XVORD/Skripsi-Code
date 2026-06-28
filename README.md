# Spatio-Temporal Behavioral Modeling for Automated Proctoring in Online Interviews

Clean research package for the final thesis and IEEE paper. The repository contains the final subject-split global model, learned-stagewise evaluation reports, paper figures, and the code needed to preprocess features, train models, evaluate modules, and generate a video demo.

The system marks video intervals for human review. It does **not** determine misconduct automatically and does not verify the use of generative AI.

## Repository Layout

```text
kode_final/
|-- code/
|   |-- src/                  # perception, temporal model, scoring, reporting
|   `-- scripts/              # preprocessing, training, evaluation, demo
|-- configs/                  # final W60 runtime/training configuration
|-- models/
|   |-- global_lstm_w60_normalized/
|   `-- object_detection/
|-- artifacts/
|   |-- global/               # final global and fusion metrics
|   |-- stagewise/            # subject-split looking/offscreen reports
|   |-- face_modules/         # face-presence and focused multiple-face tests
|   |-- object_detection/     # externally executed CITW summary
|   `-- figures/              # figures used by the IEEE paper
|-- paper/                    # compiled 11-page IEEE paper
|-- demo/                     # generic PowerShell demo launcher
|-- RESULTS.md                # auditable metric summary
`-- ARTIFACTS.md              # provenance and evaluation-unit map
```

Raw participant videos, full feature caches, reannotation drafts, obsolete W150 experiments, and legacy checkpoints are intentionally excluded. This keeps the repository small and avoids publishing participant data.

## Environment

Python 3.10 or 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run a Demo

Supply a local video; no participant video is committed to the repository.

```powershell
.\demo\run_demo.ps1 -VideoPath "D:\path\to\video.mp4"
```

Equivalent Python command:

```powershell
python code\scripts\run_main_risk_demo.py `
  --video "D:\path\to\video.mp4" `
  --config configs\config_final_w60.yaml `
  --output-dir demo\generated_outputs `
  --max-frames 450 `
  --status-source temporal
```

The output directory contains an overlay video, a JSON summary, and detected intervals in CSV format.

## Final Model

The reported global model is a one-layer attention LSTM with 64 hidden units, W60, stride 30, per-video z-score normalization, and a validation-selected threshold of 0.60. The split is participant based: 14 training, 3 validation, and 3 test subjects.

```text
models/global_lstm_w60_normalized/checkpoint_best.pth
models/global_lstm_w60_normalized/split_manifest.json
models/global_lstm_w60_normalized/test_results.json
models/global_lstm_w60_normalized/training_history.json
```

The final unseen-subject test contains 794 windows. See `RESULTS.md` and `artifacts/global/global_metrics.json` for confusion counts and metrics.

## Reproduce Training

Full reproduction requires the private feature and label directories generated from the 60 videos. Their expected layout is documented by the preprocessing scripts but the data are not included.

```powershell
python code\scripts\train_lstm.py `
  --features-dir data\processed\features `
  --labels-dir data\processed\labels `
  --split-unit subject `
  --window-size 60 `
  --stride 30 `
  --hidden-size 64 `
  --num-layers 1 `
  --dropout 0.45 `
  --label-mode majority `
  --monitor val_loss `
  --lr-scheduler cosine `
  --normalize-per-video `
  --output-dir models\reproduced_global_w60
```

Learned-stagewise evaluation uses learned W60 sequence classifiers rather than frame-level decision rules:

```powershell
python code\scripts\train_stagewise_temporal.py --help
```

## Validate the Package

```powershell
python code\scripts\verify_final_artifacts.py
pytest -q
```

The validation script recomputes metrics from the stored confusion counts and checks that the subject partitions do not overlap.
