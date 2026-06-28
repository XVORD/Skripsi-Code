"""Verify the published metrics and subject split from stored artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def binary_metrics(tn: int, fp: int, fn: int, tp: int) -> dict[str, float]:
    n = tn + fp + fn + tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": n, "accuracy": (tp + tn) / n, "precision": precision, "recall": recall, "f1": f1}


def assert_close(actual: float, expected: float, tolerance: float = 1e-8) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected:.10f}, got {actual:.10f}")


def verify_confusion_artifact(path: str) -> None:
    data = load(path)
    metrics = binary_metrics(data["tn"], data["fp"], data["fn"], data["tp"])
    if metrics["n"] != data["n"]:
        raise AssertionError(f"Sample count mismatch in {path}")
    assert_close(metrics["accuracy"], data["accuracy"])
    assert_close(metrics["precision"], data["suspicious_precision"])
    assert_close(metrics["recall"], data["suspicious_recall"])
    assert_close(metrics["f1"], data["suspicious_f1"])


def verify_split() -> None:
    split = load("models/global_lstm_w60_normalized/split_manifest.json")
    if split.get("split_unit") != "subject":
        raise AssertionError("Final model is not marked as subject split")
    train, val, test = map(set, (split["train_subjects"], split["val_subjects"], split["test_subjects"]))
    if train & val or train & test or val & test:
        raise AssertionError("Subject leakage detected between partitions")
    if (len(train), len(val), len(test)) != (14, 3, 3):
        raise AssertionError("Unexpected subject counts")


def verify_stagewise() -> None:
    cases = [
        ("artifacts/stagewise/looking_away_temporal_w60/stagewise_temporal_ablation.json", 113, 498, 98, 85),
        ("artifacts/stagewise/offscreen_gaze_temporal_w60/stagewise_temporal_ablation.json", 164, 435, 88, 107),
    ]
    for path, tp, tn, fp, fn in cases:
        data = load(path)
        result = data["results"][0]
        observed = (result["tp"], result["tn"], result["fp"], result["fn"])
        if observed != (tp, tn, fp, fn):
            raise AssertionError(f"Stagewise confusion mismatch in {path}: {observed}")
        if result["test_windows"] != 794:
            raise AssertionError(f"Unexpected test-window count in {path}")
        if not data["config"].get("split_manifest"):
            raise AssertionError(f"Missing subject split manifest in {path}")


def main() -> None:
    verify_confusion_artifact("artifacts/global/global_metrics.json")
    verify_confusion_artifact("artifacts/global/decision_support_metrics.json")
    verify_split()
    verify_stagewise()
    print("OK: final metrics, confusion counts, and subject partitions are consistent.")


if __name__ == "__main__":
    main()
