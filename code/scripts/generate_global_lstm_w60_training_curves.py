import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = (
    ROOT
    / "models"
    / "global_lstm_window_comparison_20260611"
    / "w60_s30_majority"
    / "training_history.json"
)
SUMMARY_PATH = (
    ROOT
    / "HASIL_FINAL_GLOBAL_W60.json"
)
OUTPUT_PATH = (
    ROOT
    / "Christopher_Satya_Fredella_Balakosa___Skripsi"
    / "assets"
    / "pics"
    / "GLOBAL_LSTM_W60_TRAINING_CURVES_20260612.png"
)


def main() -> None:
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    final_metrics = summary["final_test_metrics"]
    training_summary = summary["training"]
    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_loss = np.asarray(history["val_loss"], dtype=float)
    train_f1 = np.asarray(history["train_f1"], dtype=float)
    val_f1 = np.asarray(history["val_f1"], dtype=float)
    epochs = np.arange(1, len(val_loss) + 1)

    best_loss_epoch = int(np.argmin(val_loss)) + 1
    best_f1_epoch = int(np.argmax(val_f1)) + 1
    expected_epochs = int(training_summary["total_epochs"])
    if len(epochs) != expected_epochs:
        raise ValueError(f"History has {len(epochs)} epochs, but summary reports {expected_epochs}.")
    if best_loss_epoch != int(training_summary["best_validation_loss_epoch"]):
        raise ValueError("Best validation-loss epoch differs between history and summary.")
    if best_f1_epoch != int(training_summary["best_validation_macro_f1_epoch"]):
        raise ValueError("Best validation-F1 epoch differs between history and summary.")

    tick_step = 5 if len(epochs) <= 50 else 10
    epoch_ticks = np.unique(np.append([1], np.arange(tick_step, len(epochs) + 1, tick_step)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    axes[0].plot(epochs, train_loss, marker="o", markersize=3, label="Train loss", color="#2563eb")
    axes[0].plot(epochs, val_loss, marker="o", markersize=3, label="Validation loss", color="#dc2626")
    axes[0].scatter(
        best_loss_epoch,
        val_loss[best_loss_epoch - 1],
        s=65,
        color="#111827",
        zorder=5,
        label=f"Minimum validation loss (epoch {best_loss_epoch})",
    )
    axes[0].set_title("Kurva Loss Model Global LSTM W60")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_xticks(epoch_ticks)
    axes[0].legend(fontsize=8)

    axes[1].plot(epochs, train_f1 * 100, marker="o", markersize=3, label="Train macro-F1", color="#16a34a")
    axes[1].plot(epochs, val_f1 * 100, marker="o", markersize=3, label="Validation macro-F1", color="#f59e0b")
    axes[1].scatter(
        best_f1_epoch,
        val_f1[best_f1_epoch - 1] * 100,
        s=65,
        color="#111827",
        zorder=5,
        label=f"Maximum validation F1 (epoch {best_f1_epoch})",
    )
    axes[1].set_title("Kurva Macro-F1 Model Global LSTM W60")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1 (%)")
    axes[1].set_xticks(epoch_ticks)
    axes[1].set_ylim(50, 80)
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"Riwayat Pelatihan Aktual Model Global LSTM Dua Layer, W60 Stride 30 ({len(epochs)} Epoch)",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        (
            f"Test final (threshold {final_metrics['decision_threshold']:.2f}, N={summary['configuration']['test_windows']}): "
            f"Accuracy {final_metrics['accuracy'] * 100:.2f}% | "
            f"Macro-F1 {final_metrics['macro_f1'] * 100:.2f}% | "
            f"F1 suspicious {final_metrics['suspicious_f1'] * 100:.2f}%"
        ),
        ha="center",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
