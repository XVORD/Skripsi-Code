"""
Training Script untuk LSTM Cheating Detection Model
=====================================================
Melatih CheatingDetectorLSTM dari temporal_model.py
menggunakan dataset feature sequences.

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --epochs 200 --lr 0.0005 --batch-size 64
    python scripts/train_lstm.py --resume models/checkpoint_best.pth
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ERROR] PyTorch not installed. Cannot run training.")
    sys.exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("[INFO] TensorBoard not available, skipping logging")

from src.temporal_model import CheatingDetectorLSTM
from scripts.dataset import create_dataloaders, LABEL_MAP, LABEL_NAMES, CheatingDetectionDataset
from scripts.dataset import split_videos


def extract_subject_id(video_stem: str) -> str:
    """Extract participant id such as p01 from cached video stem."""
    import re
    m = re.search(r"(p\d+)_vid_", str(video_stem).lower())
    if not m:
        raise ValueError(f"Cannot extract subject id from video stem: {video_stem}")
    return m.group(1)

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        f1_score, accuracy_score, roc_auc_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def apply_profile(args):
    """Apply named experiment profile to align training settings."""
    profile = str(getattr(args, "profile", "none")).strip().lower()
    if profile in ("", "none"):
        return args

    if profile == "thesis_baseline":
        args.window_size = 30
        args.stride = 15
        args.hidden_size = 128
        args.num_layers = 2
        args.dropout = 0.3
        args.label_mode = "any_suspicious"
        return args

    if profile == "thesis_stable":
        args.window_size = 30
        args.stride = 15
        args.hidden_size = 128
        args.num_layers = 2
        args.dropout = 0.3
        args.label_mode = "any_suspicious"
        args.monitor = "val_loss"
        args.lr_scheduler = "cosine"
        args.lr = min(float(args.lr), 8e-4)
        args.min_lr = max(float(args.min_lr), 1e-5)
        args.weight_decay = max(float(args.weight_decay), 8e-4)
        args.label_smoothing = max(float(args.label_smoothing), 0.05)
        args.augment_noise_std = max(float(args.augment_noise_std), 0.02)
        args.augment_mask_prob = max(float(args.augment_mask_prob), 0.35)
        args.augment_mask_max_frac = max(float(args.augment_mask_max_frac), 0.30)
        args.augment_feature_dropout_prob = max(float(args.augment_feature_dropout_prob), 0.10)
        args.min_epochs = max(int(args.min_epochs), 20)
        args.overfit_guard_patience = max(int(args.overfit_guard_patience), 6)
        args.overfit_guard_tol = min(float(args.overfit_guard_tol), 0.02)
        args.patience = min(int(args.patience), 20)
        return args

    raise ValueError(f"Unknown profile: {args.profile}")


def set_global_seed(seed: int):
    """Set deterministic seeds for repeatable experiments."""
    s = int(seed)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    threshold: float = 0.5,
                    label_smoothing: float = 0.0,
                    scheduler=None,
                    scheduler_per_batch: bool = False):
    """Train model untuk satu epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).float()
        if label_smoothing > 0:
            s = float(max(0.0, min(0.49, label_smoothing)))
            labels = labels * (1.0 - s) + 0.5 * s

        optimizer.zero_grad()
        logits = model(features).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None and scheduler_per_batch:
            scheduler.step()

        total_loss += loss.item() * features.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.long().cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluasi model pada validation/test set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).float()

        logits = model(features).squeeze(1)
        loss = criterion(logits, labels)

        total_loss += loss.item() * features.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.long().cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, f1, all_preds, all_labels, np.array(all_probs)


@torch.no_grad()
def evaluate_with_threshold(model, dataloader, criterion, device, threshold: float = 0.5):
    """Evaluasi model dengan decision threshold tertentu."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).float()

        logits = model(features).squeeze(1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += loss.item() * features.size(0)
        all_labels.extend(labels.long().cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = (all_probs >= float(threshold)).astype(np.int64)
    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels, all_probs


def tune_threshold(y_true, y_probs, min_thr=0.10, max_thr=0.90, step=0.01):
    """Cari threshold terbaik berdasarkan macro-F1 pada validation set."""
    best_thr = 0.5
    best_f1 = -1.0
    thresholds = np.arange(min_thr, max_thr + 1e-9, step)
    for thr in thresholds:
        preds = (y_probs >= thr).astype(np.int64)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def print_classification_report(labels, preds, probs=None):
    """Print classification report dan confusion matrix."""
    if not SKLEARN_AVAILABLE:
        print("[WARNING] scikit-learn not available for detailed report")
        return

    # Classification report
    target_names = [LABEL_NAMES[i] for i in range(len(LABEL_MAP))]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        labels, preds, labels=[0, 1], target_names=target_names, zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print("CONFUSION MATRIX:")
    print(f"{'':>15}", end="")
    for name in target_names:
        print(f"{name[:10]:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{target_names[i][:15]:>15}", end="")
        for val in row:
            print(f"{val:>12d}", end="")
        print()

    # AUC-ROC
    if probs is not None:
        try:
            auc = roc_auc_score(labels, probs)
            print(f"\nROC-AUC: {auc:.4f}")
        except Exception as e:
            print(f"\n[WARNING] Could not compute AUC-ROC: {e}")


def train(args):
    """Main training function."""
    args = apply_profile(args)
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"LSTM Cheating Detection Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Window size: {args.window_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Num layers: {args.num_layers}")
    print(f"Dropout: {args.dropout}")
    print(f"Profile: {args.profile}")
    print(f"LR scheduler: {args.lr_scheduler}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("[1/4] Loading dataset...")
    train_videos, val_videos, test_videos = split_videos(
        features_dir=args.features_dir,
        labels_dir=args.labels_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_unit=args.split_unit,
    )
    split_manifest = {
        "seed": int(args.seed),
        "split_unit": str(args.split_unit),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(max(0.0, 1.0 - args.train_ratio - args.val_ratio)),
        "label_mode": args.label_mode,
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "train_subjects": sorted({extract_subject_id(v) for v in train_videos}),
        "val_subjects": sorted({extract_subject_id(v) for v in val_videos}),
        "test_subjects": sorted({extract_subject_id(v) for v in test_videos}),
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
    }
    with open(str(output_dir / "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        features_dir=args.features_dir,
        labels_dir=args.labels_dir,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        label_mode=args.label_mode,
        augment_noise_std=args.augment_noise_std,
        augment_mask_prob=args.augment_mask_prob,
        augment_mask_max_frac=args.augment_mask_max_frac,
        augment_feature_dropout_prob=args.augment_feature_dropout_prob,
        augment_time_reverse_prob=args.augment_time_reverse_prob,
        video_splits=(train_videos, val_videos, test_videos),
        split_unit=args.split_unit,
        normalize_per_video=args.normalize_per_video,
    )
    # Clean train loader (no augmentation) for fair train-vs-val curve comparison.
    pin_memory = torch.cuda.is_available()
    train_eval_ds = CheatingDetectionDataset(
        args.features_dir, args.labels_dir,
        window_size=args.window_size,
        stride=args.stride,
        augment=False,
        video_list=train_videos,
        label_mode=args.label_mode,
        normalize_per_video=args.normalize_per_video,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        train_eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    # Model
    print("\n[2/4] Building model...")
    model = CheatingDetectorLSTM(
        input_size=args.num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_outputs=1,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss, optimizer, scheduler
    train_samples = getattr(train_loader.dataset, "samples", [])
    pos_count = sum(int(s["label"]) for s in train_samples) if train_samples else 0
    neg_count = max(len(train_samples) - pos_count, 0)
    pos_weight = float(neg_count / max(pos_count, 1))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    elif args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(args.epochs)),
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=max(1, int(args.epochs)),
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
            anneal_strategy="cos",
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    if args.monitor == "val_loss":
        best_monitor_value = float("inf")
    else:
        best_monitor_value = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"[WARNING] Could not load scheduler state: {e}")
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("best_val_f1", 0.0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        if args.monitor == "val_loss":
            best_monitor_value = checkpoint.get("best_monitor_value", best_val_loss)
        else:
            best_monitor_value = checkpoint.get("best_monitor_value", best_val_f1)
        print(f"Resumed from epoch {start_epoch}, best val F1: {best_val_f1:.4f}")

    # TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE and not args.no_tensorboard:
        log_dir = output_dir / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard logs: {log_dir}")

    # Training loop
    print("\n[3/4] Training...")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": [], "lr": [],
               "train_loss_clean": [], "train_acc_clean": [], "train_f1_clean": []}
    patience_counter = 0
    overfit_counter = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            label_smoothing=args.label_smoothing,
            scheduler=scheduler if args.lr_scheduler == "onecycle" else None,
            scheduler_per_batch=(args.lr_scheduler == "onecycle")
        )

        # Validate
        val_loss, val_acc, val_f1, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        train_loss_clean, train_acc_clean, train_f1_clean, _, _, _ = evaluate(
            model, train_eval_loader, criterion, device
        )

        # Scheduler
        if args.lr_scheduler == "plateau":
            scheduler.step(val_loss)
        elif args.lr_scheduler == "cosine":
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # History
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        history["train_loss_clean"].append(train_loss_clean)
        history["train_acc_clean"].append(train_acc_clean)
        history["train_f1_clean"].append(train_f1_clean)

        elapsed = time.time() - t0

        # Log
        if (epoch + 1) % args.log_interval == 0 or epoch == 0:
            print(f"Epoch {epoch+1:>4d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                  f"TrainClean Loss: {train_loss_clean:.4f} Acc: {train_acc_clean:.4f} F1: {train_f1_clean:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                  f"LR: {current_lr:.6f} | {elapsed:.1f}s")

        # TensorBoard
        if writer:
            writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
            writer.add_scalars("F1", {"train": train_f1, "val": val_f1}, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Save best model by selected monitor metric
        if args.monitor == "val_loss":
            current_monitor_value = val_loss
            improved = current_monitor_value < best_monitor_value
        else:
            current_monitor_value = val_f1
            improved = current_monitor_value > best_monitor_value

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if improved:
            best_monitor_value = current_monitor_value
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1,
                "best_val_loss": best_val_loss,
                "best_monitor_value": best_monitor_value,
                "monitor_metric": args.monitor,
                "lr_scheduler": args.lr_scheduler,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "config": {
                    "input_size": args.num_features,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "num_outputs": 1,
                    "dropout": args.dropout,
                    "window_size": args.window_size,
                    "label_mode": args.label_mode,
                },
            }
            torch.save(checkpoint, str(output_dir / "checkpoint_best.pth"))
            torch.save(model.state_dict(), str(output_dir / "cheating_detector_lstm.pth"))
            print(f"  [OK] Best model saved ({args.monitor}={best_monitor_value:.4f}, "
                  f"best_f1={best_val_f1:.4f}, best_val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Overfit guard: stop if val_loss keeps drifting above best_val_loss.
        if args.monitor == "val_loss":
            if val_loss > (best_val_loss * (1.0 + float(args.overfit_guard_tol))):
                overfit_counter += 1
            else:
                overfit_counter = 0

            if (epoch + 1) >= int(args.min_epochs) and overfit_counter >= int(args.overfit_guard_patience):
                print(f"\n[OVERFIT GUARD] val_loss stayed > best_val_loss by "
                      f"{args.overfit_guard_tol*100:.1f}% for {overfit_counter} epochs. "
                      f"Stopping at epoch {epoch+1}.")
                break

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[EARLY STOP] No improvement for {args.patience} epochs")
            break

    # Save final model
    torch.save(model.state_dict(), str(output_dir / "cheating_detector_lstm_final.pth"))

    # Save history
    with open(str(output_dir / "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Test evaluation (load best checkpoint first)
    print("\n[4/4] Evaluating on test set...")
    best_ckpt_path = output_dir / "checkpoint_best.pth"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(str(best_ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state_dict"])

    # Threshold tuning on validation (to improve suspicious recall/F1 balance)
    _, _, _, _, val_labels, val_probs = evaluate_with_threshold(
        model, val_loader, criterion, device, threshold=0.5
    )
    decision_threshold, val_f1_tuned = tune_threshold(
        np.array(val_labels),
        np.array(val_probs),
        min_thr=args.threshold_min,
        max_thr=args.threshold_max,
        step=args.threshold_step,
    )
    print(f"[INFO] Tuned decision threshold (val macro-F1): {decision_threshold:.2f} "
          f"(val F1={val_f1_tuned:.4f})")

    test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = evaluate_with_threshold(
        model, test_loader, criterion, device, threshold=decision_threshold
    )
    print(f"\nTest Results: Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | "
          f"F1: {test_f1:.4f} | Thr: {decision_threshold:.2f}")
    print_classification_report(test_labels, test_preds, test_probs)

    # Save test results
    test_results = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "best_val_f1": best_val_f1,
        "best_val_loss": best_val_loss,
        "monitor_metric": args.monitor,
        "best_monitor_value": float(best_monitor_value),
        "decision_threshold": float(decision_threshold),
        "val_f1_tuned_threshold": float(val_f1_tuned),
        "total_epochs": epoch + 1,
        "label_mode": args.label_mode,
    }
    with open(str(output_dir / "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    if writer:
        writer.close()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model saved to: {output_dir / 'cheating_detector_lstm.pth'}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM Cheating Detection Model"
    )

    # Data
    parser.add_argument("--features-dir", type=str, default="data/processed/features")
    parser.add_argument("--labels-dir", type=str, default="data/processed/labels")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-unit", type=str, default="subject",
                        choices=["video", "subject"],
                        help="Split by individual videos or by participant/subject")
    parser.add_argument("--label-mode", type=str, default="any_suspicious",
                        choices=["majority", "any_suspicious", "all_suspicious"])

    # Model
    parser.add_argument("--num-features", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--stride", type=int, default=15)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--threshold-min", type=float, default=0.10)
    parser.add_argument("--threshold-max", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--profile", type=str, default="none",
                        choices=["none", "thesis_baseline", "thesis_stable"],
                        help="Preset config profile")
    parser.add_argument("--monitor", type=str, default="val_f1",
                        choices=["val_f1", "val_loss"],
                        help="Metric for best-checkpoint and early-stopping monitor")
    parser.add_argument("--lr-scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine", "onecycle"],
                        help="Learning rate scheduler strategy")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate floor for scheduler")
    parser.add_argument("--plateau-factor", type=float, default=0.5,
                        help="LR decay factor for ReduceLROnPlateau")
    parser.add_argument("--plateau-patience", type=int, default=10,
                        help="Patience epochs for ReduceLROnPlateau")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3,
                        help="OneCycleLR warmup percentage")
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0,
                        help="OneCycleLR initial_lr = max_lr/div_factor")
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10000.0,
                        help="OneCycleLR min_lr = initial_lr/final_div_factor")
    parser.add_argument("--min-epochs", type=int, default=8,
                        help="Minimum epochs before overfit guard can stop training")
    parser.add_argument("--overfit-guard-patience", type=int, default=4,
                        help="Consecutive epochs of degraded val_loss before forced stop")
    parser.add_argument("--overfit-guard-tol", type=float, default=0.05,
                        help="Relative tolerance above best_val_loss for overfit guard")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing for BCE targets (0.0-0.49)")
    parser.add_argument("--augment-noise-std", type=float, default=0.01,
                        help="Gaussian noise std for train-time feature augmentation")
    parser.add_argument("--augment-mask-prob", type=float, default=0.30,
                        help="Probability of random temporal masking per sample")
    parser.add_argument("--augment-mask-max-frac", type=float, default=0.25,
                        help="Max fraction of window length to mask")
    parser.add_argument("--augment-feature-dropout-prob", type=float, default=0.0,
                        help="Probability of dropping each feature channel during augmentation")

    # Output
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--augment-time-reverse-prob", type=float, default=0.0,
                        help="Prob time reversal augmentation")
    parser.add_argument("--normalize-per-video", action="store_true",
                        help="Per-video z-score normalization untuk menghilangkan bias absolut antar subjek")

    args = parser.parse_args()
    train(args)
