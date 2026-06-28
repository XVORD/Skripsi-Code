import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "HASIL_FINAL_GLOBAL_W60.json"
OUTPUT_PATH = (
    ROOT
    / "Christopher_Satya_Fredella_Balakosa___Skripsi"
    / "assets"
    / "pics"
    / "GLOBAL_LSTM_W60_FINAL_CONFUSION_MATRIX_20260612.png"
)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    metrics = summary["final_test_metrics"]
    matrix = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
    )

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(matrix, cmap="Blues")

    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() * 0.55 else "#111827"
            ax.text(col, row, f"{value}", ha="center", va="center", fontsize=20, fontweight="bold", color=color)

    ax.set_xticks([0, 1], ["Prediksi Normal", "Prediksi Suspicious"])
    ax.set_yticks([0, 1], ["Aktual Normal", "Aktual Suspicious"])
    ax.set_xlabel("Prediksi Model")
    ax.set_ylabel("Label Aktual")
    ax.set_title("Confusion Matrix Model Global LSTM W60", fontsize=15, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.text(
        0.5,
        0.02,
        (
            f"Threshold {metrics['decision_threshold']:.2f} | N={matrix.sum()} | "
            f"Accuracy {metrics['accuracy'] * 100:.2f}% | Macro-F1 {metrics['macro_f1'] * 100:.2f}%"
        ),
        ha="center",
        fontsize=10,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
