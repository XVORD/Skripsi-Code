"""
Target-specific temporal ablation for stagewise behaviours.

This experiment answers the advisor's question:
"Why are offscreen_gaze and looking_away not learned temporally?"

It builds per-frame labels for offscreen_gaze and looking_away from the
stagewise segment template, trains learned temporal classifiers, and compares
them against the existing rule-derived stagewise signal as a baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.dataset import split_videos
from src.temporal_model import CheatingDetectorLSTM


TARGETS = {
    "offscreen_gaze": {
        "gt_column": "offscreen_gaze_gt",
        "rule_feature_index": 6,
    },
    "looking_away": {
        "gt_column": "looking_away_gt",
        "rule_feature_index": 5,
    },
}

NUM_FEATURES = 15


def masked_except(kept_indices: Sequence[int]) -> List[int]:
    kept = set(int(i) for i in kept_indices)
    return [i for i in range(NUM_FEATURES) if i not in kept]


FORWARD_FEATURE_STEPS = {
    "looking_away": [
        ("forward_look_1_pose", "Pose", [0, 1, 2]),
        ("forward_look_2_add_motion", "PoseMotion", [0, 1, 2, 13]),
        ("forward_look_3_add_gaze", "PoseMotionGaze", [0, 1, 2, 3, 4, 13, 14]),
        ("forward_look_4_add_eye_open", "AddEyeOpen", [0, 1, 2, 3, 4, 11, 12, 13, 14]),
        ("forward_look_5_add_events", "AddEvents", [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14]),
        ("forward_look_6_all", "AllFeatures", list(range(NUM_FEATURES))),
    ],
    "offscreen_gaze": [
        ("forward_off_1_gaze", "Gaze", [3, 4]),
        ("forward_off_2_add_motion", "GazeMotion", [3, 4, 14]),
        ("forward_off_3_add_eye_open", "AddEyeOpen", [3, 4, 11, 12, 14]),
        ("forward_off_4_add_pose", "AddPose", [0, 1, 2, 3, 4, 11, 12, 13, 14]),
        ("forward_off_5_add_events", "AddEvents", [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14]),
        ("forward_off_6_all", "AllFeatures", list(range(NUM_FEATURES))),
    ],
}

FEATURE_GROUPS = {
    "full": [],
    "no_gaze": [3, 4, 6, 7, 8, 14],
    "no_head_pose": [0, 1, 2, 5, 13],
    "no_eye_open": [11, 12],
    "gaze_only": [0, 1, 2, 5, 9, 10, 13],
    "head_pose_only": [3, 4, 6, 7, 8, 9, 10, 11, 12, 14],
    # Strict learned setting for advisor-facing stagewise experiments:
    # remove binary stagewise/rule outputs and keep continuous detector cues.
    "no_stagewise_rules": [5, 6, 10],
    "continuous_only": [5, 6, 7, 8, 10],
}
for _steps in FORWARD_FEATURE_STEPS.values():
    for _mask, _, _kept in _steps:
        FEATURE_GROUPS[_mask] = masked_except(_kept)

HARD_NEGATIVE_CONTINUOUS_FEATURES = [0, 1, 2, 3, 4, 11, 12, 13, 14]

# Legacy annotation filenames were produced before the final pXX naming pass.
# This mapping is intentionally explicit and exported with every experiment.
# Public training data must use canonical participant IDs (for example, p03).
# Private name-to-ID aliases are intentionally excluded from this repository.
LEGACY_NAME_TO_CANONICAL: dict[str, str] = {}

SPECIAL_VIDEO_ALIASES = {
    "normal/p01_vid_003.mp4": "normal/p10_vid_010.mp4",
    "third_party/p01_vid_002.mp4": "third_party/p10_vid_010.mp4",
}


class WindowDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[np.ndarray, int]],
        masked_indices: Sequence[int] | None = None,
        augment: bool = False,
        noise_std: float = 0.01,
        time_mask_prob: float = 0.10,
        feature_dropout_prob: float = 0.0,
        feature_scale_std: float = 0.0,
        time_jitter_prob: float = 0.0,
        time_jitter_max: int = 0,
        temporal_smooth_prob: float = 0.0,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        self.samples = list(samples)
        self.masked_indices = list(masked_indices or [])
        self.augment = bool(augment)
        self.noise_std = float(noise_std)
        self.time_mask_prob = float(time_mask_prob)
        self.feature_dropout_prob = float(feature_dropout_prob)
        self.feature_scale_std = float(feature_scale_std)
        self.time_jitter_prob = float(time_jitter_prob)
        self.time_jitter_max = int(time_jitter_max)
        self.temporal_smooth_prob = float(temporal_smooth_prob)
        self.feature_mean = None if feature_mean is None else torch.tensor(feature_mean, dtype=torch.float32)
        self.feature_std = None if feature_std is None else torch.tensor(feature_std, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        features, label = self.samples[index]
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.tensor(features, dtype=torch.float32)
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std
        valid_mask = [idx for idx in self.masked_indices if idx < x.shape[1]]
        if valid_mask:
            x[:, valid_mask] = 0.0
        if self.augment:
            if self.feature_scale_std > 0:
                scale = 1.0 + torch.randn(x.shape[1], dtype=torch.float32) * self.feature_scale_std
                x = x * scale
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.time_jitter_prob > 0 and self.time_jitter_max > 0 and random.random() < self.time_jitter_prob:
                shift = random.randint(-self.time_jitter_max, self.time_jitter_max)
                if shift != 0:
                    x = torch.roll(x, shifts=shift, dims=0)
            if self.feature_dropout_prob > 0 and random.random() < self.feature_dropout_prob:
                keep = torch.ones(x.shape[1], dtype=torch.float32)
                n_drop = random.randint(1, max(1, x.shape[1] // 5))
                drop_idx = torch.randperm(x.shape[1])[:n_drop]
                keep[drop_idx] = 0.0
                x = x * keep
            if random.random() < self.time_mask_prob and x.shape[0] > 4:
                max_len = max(1, x.shape[0] // 5)
                mask_len = random.randint(1, max_len)
                start = random.randint(0, x.shape[0] - mask_len)
                x[start : start + mask_len] = 0.0
            if self.temporal_smooth_prob > 0 and x.shape[0] > 2 and random.random() < self.temporal_smooth_prob:
                smoothed = x.clone()
                smoothed[1:-1] = 0.25 * x[:-2] + 0.50 * x[1:-1] + 0.25 * x[2:]
                x = smoothed
        y = torch.tensor(label, dtype=torch.long)
        if valid_mask:
            x[:, valid_mask] = 0.0
        return x, y


class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 15,
        seq_len: int = 30,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        z = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        z = self.encoder(z)
        return self.classifier(z.mean(dim=1))


class TemporalConvClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        z = x.transpose(1, 2)
        return self.classifier(self.net(z))


class LSTMStatsClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 15,
        hidden_size: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        stats_size = input_size * 5
        self.stats_proj = nn.Sequential(
            nn.LayerNorm(stats_size),
            nn.Linear(stats_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        z, _ = self.lstm(x)
        weights = torch.softmax(self.attn(z), dim=1)
        pooled = torch.sum(z * weights, dim=1)
        stats = torch.cat(
            [
                x.mean(dim=1),
                x.std(dim=1, unbiased=False),
                x.amin(dim=1),
                x.amax(dim=1),
                x[:, -1, :] - x[:, 0, :],
            ],
            dim=1,
        )
        stats = self.stats_proj(stats)
        return self.classifier(torch.cat([pooled, stats], dim=1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def normalize_video_key(video: str) -> str:
    return str(video).replace("\\", "/").strip()


def canonical_video(video: str) -> str:
    video = normalize_video_key(video)
    if video in SPECIAL_VIDEO_ALIASES:
        return SPECIAL_VIDEO_ALIASES[video]

    scenario, filename = video.split("/", 1)
    stem = filename.rsplit(".", 1)[0]
    if stem.startswith("p") and "_vid_" in stem:
        # Already canonical pXX naming.
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0][1:].isdigit():
            pnum = int(parts[0][1:])
            if 1 <= pnum <= 20 and len(parts[0]) == 3:
                return f"{scenario}/{parts[0]}_vid_{pnum:03d}.mp4"

    legacy_name = stem.rsplit("_vid_", 1)[0]
    mapped = LEGACY_NAME_TO_CANONICAL.get(legacy_name)
    if mapped:
        return f"{scenario}/{mapped}.mp4"
    return video


def video_to_stem(video: str) -> str:
    video = canonical_video(video)
    scenario, filename = video.split("/", 1)
    stem = filename.rsplit(".", 1)[0]
    return f"{scenario}__{stem}"


def stem_to_video(stem: str) -> str:
    scenario, rest = stem.split("__", 1)
    return f"{scenario}/{rest}.mp4"


def to_int(value) -> int | None:
    try:
        if value is None:
            return None
        if str(value).strip() == "":
            return None
        return int(float(value))
    except Exception:
        return None


def load_segments(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_target_labels(
    segments_path: Path,
    features_dir: Path,
    target: str,
    output_dir: Path,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[dict]]:
    target_info = TARGETS[target]
    gt_col = target_info["gt_column"]
    feature_files = {p.stem: p for p in features_dir.glob("*.npy")}
    labels: Dict[str, np.ndarray] = {}
    label_known: Dict[str, np.ndarray] = {}
    mapping_rows: List[dict] = []

    for stem, feature_path in sorted(feature_files.items()):
        n = int(np.load(str(feature_path), mmap_mode="r").shape[0])
        labels[stem] = np.zeros(n, dtype=np.int64)
        label_known[stem] = np.zeros(n, dtype=np.bool_)

    for row in load_segments(segments_path):
        original = normalize_video_key(row.get("video", ""))
        canonical = canonical_video(original)
        stem = video_to_stem(canonical)
        if stem not in labels:
            mapping_rows.append(
                {
                    "original_video": original,
                    "canonical_video": canonical,
                    "canonical_stem": stem,
                    "status": "missing_feature_file",
                }
            )
            continue
        y = to_int(row.get(gt_col))
        if y not in (0, 1):
            continue
        start = to_int(row.get("start_frame"))
        end = to_int(row.get("end_frame"))
        if start is None or end is None:
            continue
        n = len(labels[stem])
        start = max(0, min(int(start), n - 1))
        end = max(start, min(int(end), n - 1))
        labels[stem][start : end + 1] = int(y)
        label_known[stem][start : end + 1] = True
        mapping_rows.append(
            {
                "original_video": original,
                "canonical_video": canonical,
                "canonical_stem": stem,
                "status": "mapped",
                "start_frame": start,
                "end_frame": end,
                "target": target,
                "label": y,
            }
        )

    # Keep only videos with any segment coverage.
    filtered = {stem: arr for stem, arr in labels.items() if bool(label_known[stem].any())}
    filtered_known = {stem: arr for stem, arr in label_known.items() if bool(label_known[stem].any())}

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{target}_label_mapping.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "original_video",
            "canonical_video",
            "canonical_stem",
            "status",
            "start_frame",
            "end_frame",
            "target",
            "label",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in mapping_rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    return filtered, filtered_known, mapping_rows


def window_label(labels: np.ndarray, mode: str, positive_ratio: float = 0.30) -> int:
    y = (labels.astype(np.int64) > 0).astype(np.int64)
    if mode == "any_positive":
        return int(np.any(y == 1))
    if mode == "all_positive":
        return int(np.all(y == 1))
    if mode == "positive_ratio":
        return int(float(np.mean(y)) >= float(positive_ratio))
    return int(np.bincount(y, minlength=2).argmax())


def make_samples(
    features_dir: Path,
    labels_by_video: Dict[str, np.ndarray],
    known_by_video: Dict[str, np.ndarray] | None,
    video_list: Iterable[str],
    window_size: int,
    stride: int,
    label_mode: str,
    positive_ratio: float = 0.30,
    min_known_ratio: float = 0.0,
) -> Tuple[List[Tuple[np.ndarray, int]], List[dict]]:
    samples: List[Tuple[np.ndarray, int]] = []
    meta: List[dict] = []
    for stem in video_list:
        if stem not in labels_by_video:
            continue
        feature_path = features_dir / f"{stem}.npy"
        if not feature_path.exists():
            continue
        features = np.load(str(feature_path)).astype(np.float32)
        labels = labels_by_video[stem].astype(np.int64)
        n = min(len(features), len(labels))
        if n < window_size:
            continue
        features = features[:n]
        labels = labels[:n]
        if known_by_video is None:
            known = np.ones(n, dtype=np.bool_)
        else:
            known = known_by_video.get(stem, np.zeros(n, dtype=np.bool_))[:n].astype(np.bool_)
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            window_known = known[start:end]
            known_ratio = float(np.mean(window_known)) if len(window_known) else 0.0
            if known_ratio < float(min_known_ratio):
                continue
            label_values = labels[start:end][window_known]
            if len(label_values) == 0:
                continue
            y = window_label(label_values, label_mode, positive_ratio=positive_ratio)
            samples.append((features[start:end], y))
            meta.append(
                {
                    "video": stem,
                    "start_frame": start,
                    "end_frame": end - 1,
                    "label": y,
                    "known_ratio": known_ratio,
                }
            )
    return samples, meta


def binary_metrics(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_pos = 2 * precision * recall / max(precision + recall, 1e-12)
    precision0 = tn / max(tn + fn, 1)
    recall0 = tn / max(tn + fp, 1)
    f1_neg = 2 * precision0 * recall0 / max(precision0 + recall0, 1e-12)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_positive": f1_pos,
        "f1_macro": (f1_pos + f1_neg) / 2.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
    }


def tune_threshold(y_true, y_prob, metric_name: str = "f1_macro") -> Tuple[float, float]:
    best_thr = 0.50
    best_score = -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        m = binary_metrics(y_true, y_prob, float(thr))
        score = float(m.get(metric_name, m["f1_macro"]))
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def smooth_probabilities_by_video(y_prob, meta: Sequence[dict] | None, span: int) -> np.ndarray:
    probs = np.asarray(y_prob, dtype=np.float32).copy()
    if span <= 1 or not meta:
        return probs
    out = probs.copy()
    radius = max(0, int(span) // 2)
    by_video: Dict[str, List[int]] = {}
    for i, row in enumerate(meta):
        by_video.setdefault(str(row.get("video", "")), []).append(i)
    for indices in by_video.values():
        indices = sorted(indices, key=lambda i: int(meta[i].get("start_frame", i)))
        vals = probs[indices]
        for local_idx, global_idx in enumerate(indices):
            lo = max(0, local_idx - radius)
            hi = min(len(indices), local_idx + radius + 1)
            out[global_idx] = float(np.mean(vals[lo:hi]))
    return out


def hard_negative_score(
    features: np.ndarray,
    masked_indices: Sequence[int],
    feature_mean: np.ndarray | None,
    feature_std: np.ndarray | None,
) -> float:
    x = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if feature_mean is not None and feature_std is not None:
        x = (x - feature_mean) / feature_std
    if masked_indices:
        x[:, list(masked_indices)] = 0.0
    cols = [
        idx
        for idx in HARD_NEGATIVE_CONTINUOUS_FEATURES
        if idx not in set(masked_indices) and idx < x.shape[1]
    ]
    if not cols:
        cols = [idx for idx in range(x.shape[1]) if idx not in set(masked_indices)]
    if not cols:
        return 0.0
    view = x[:, cols]
    # High absolute pose/gaze movement plus temporal variation marks difficult normal windows.
    return float(np.mean(np.abs(view)) + 0.50 * np.mean(np.std(view, axis=0)))


def balance_train_samples(
    samples: Sequence[Tuple[np.ndarray, int]],
    positive_multiplier: int,
    hard_negative_multiplier: int = 1,
    hard_negative_top_frac: float = 0.25,
    masked_indices: Sequence[int] | None = None,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> List[Tuple[np.ndarray, int]]:
    samples = list(samples)
    positives = [s for s in samples if int(s[1]) == 1]
    negatives = [s for s in samples if int(s[1]) == 0]
    if not positives or not negatives:
        return samples
    balanced = list(samples)
    for _ in range(max(0, int(positive_multiplier) - 1)):
        balanced.extend(positives)
    if hard_negative_multiplier > 1:
        masked_indices = list(masked_indices or [])
        scored = [
            (
                hard_negative_score(x, masked_indices, feature_mean, feature_std),
                (x, y),
            )
            for x, y in negatives
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        keep_n = max(1, int(math.ceil(len(scored) * float(hard_negative_top_frac))))
        hard_negatives = [sample for _, sample in scored[:keep_n]]
        for _ in range(max(0, int(hard_negative_multiplier) - 1)):
            balanced.extend(hard_negatives)
    random.shuffle(balanced)
    return balanced


def run_eval(model, loader, criterion, device):
    model.eval()
    y_true = []
    y_prob = []
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            yf = y.to(device).float()
            logits = model(x).squeeze(1)
            loss = criterion(logits, yf)
            total_loss += float(loss.item()) * x.shape[0]
            y_true.extend(y.cpu().numpy().astype(int).tolist())
            y_prob.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return total_loss / max(len(loader.dataset), 1), np.array(y_true), np.array(y_prob)


def build_prediction_rows(
    variant: dict,
    meta: Sequence[dict] | None,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> List[dict]:
    rows: List[dict] = []
    meta = list(meta or [])
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= float(threshold)).astype(int)
    for i, (gt, pred, prob) in enumerate(zip(y_true, y_pred, y_prob)):
        item = meta[i] if i < len(meta) else {}
        if gt == 1 and pred == 1:
            result = "TP"
        elif gt == 0 and pred == 0:
            result = "TN"
        elif gt == 0 and pred == 1:
            result = "FP"
        else:
            result = "FN"
        start_frame = int(item.get("start_frame", 0))
        end_frame = int(item.get("end_frame", start_frame))
        rows.append(
            {
                "target": variant["target"],
                "model": variant["name"],
                "video": item.get("video", ""),
                "video_path": stem_to_video(str(item.get("video", ""))) if item.get("video", "") else "",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": round(start_frame / 15.0, 3),
                "end_sec": round(end_frame / 15.0, 3),
                "window_size": int(variant["window_size"]),
                "window_seconds": round(int(variant["window_size"]) / 15.0, 3),
                "stride": int(variant["stride"]),
                "known_ratio": float(item.get("known_ratio", 1.0)),
                "gt_label": int(gt),
                "pred_label": int(pred),
                "probability": float(prob),
                "threshold": float(threshold),
                "result": result,
                "corrected_label": "",
                "review_notes": "",
            }
        )
    return rows


def train_model(
    args,
    variant: dict,
    train_samples,
    val_samples,
    test_samples,
    train_meta: Sequence[dict] | None = None,
    val_meta: Sequence[dict] | None = None,
    test_meta: Sequence[dict] | None = None,
) -> Tuple[dict, list]:
    set_seed(args.model_seed if args.model_seed is not None else args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    window_size = int(variant["window_size"])
    input_size = int(train_samples[0][0].shape[1])
    if variant.get("mask") == "no_target_rule":
        masked_indices = [TARGETS[variant["target"]]["rule_feature_index"]]
    else:
        masked_indices = FEATURE_GROUPS[variant.get("mask", "full")]

    if args.no_standardize:
        feature_mean = None
        feature_std = None
    else:
        flat_train = np.concatenate([np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x, _ in train_samples], axis=0)
        feature_mean = flat_train.mean(axis=0).astype(np.float32)
        feature_std = flat_train.std(axis=0).astype(np.float32)
        feature_std[feature_std < 1e-6] = 1.0
    train_samples_for_loader = balance_train_samples(
        train_samples,
        positive_multiplier=args.positive_oversample,
        hard_negative_multiplier=args.hard_negative_oversample,
        hard_negative_top_frac=args.hard_negative_top_frac,
        masked_indices=masked_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )

    train_loader = DataLoader(
        WindowDataset(
            train_samples_for_loader,
            masked_indices,
            augment=True,
            noise_std=args.noise_std,
            time_mask_prob=args.time_mask_prob,
            feature_dropout_prob=args.feature_dropout_prob,
            feature_scale_std=args.feature_scale_std,
            time_jitter_prob=args.time_jitter_prob,
            time_jitter_max=args.time_jitter_max,
            temporal_smooth_prob=args.temporal_smooth_prob,
            feature_mean=feature_mean,
            feature_std=feature_std,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WindowDataset(val_samples, masked_indices, feature_mean=feature_mean, feature_std=feature_std),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WindowDataset(test_samples, masked_indices, feature_mean=feature_mean, feature_std=feature_std),
        batch_size=args.batch_size,
        shuffle=False,
    )
    train_eval_loader = DataLoader(
        WindowDataset(train_samples, masked_indices, feature_mean=feature_mean, feature_std=feature_std),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if variant["model"] == "lstm":
        model = CheatingDetectorLSTM(
            input_size=input_size,
            hidden_size=args.hidden_size,
            num_layers=2,
            num_outputs=1,
            dropout=args.dropout,
        )
    elif variant["model"] == "transformer":
        model = SimpleTransformerClassifier(
            input_size=input_size,
            seq_len=window_size,
            d_model=args.transformer_dim,
            nhead=args.transformer_heads,
            num_layers=args.transformer_layers,
            dropout=args.dropout,
        )
    elif variant["model"] == "temporal_conv":
        model = TemporalConvClassifier(
            input_size=input_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    elif variant["model"] == "lstm_stats":
        model = LSTMStatsClassifier(
            input_size=input_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"Unknown learned model: {variant['model']}")

    labels = [int(y) for _, y in train_samples]
    pos = int(sum(labels))
    neg = int(len(labels) - pos)
    pos_weight = torch.tensor(neg / max(pos, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = model.to(device)

    best_state = None
    best_val_f1 = -1.0
    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            yf = y.to(device).float()
            if args.label_smoothing > 0:
                eps = float(args.label_smoothing)
                yf = yf * (1.0 - 2.0 * eps) + eps
            optimizer.zero_grad()
            logits = model(x).squeeze(1)
            loss = criterion(logits, yf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.item()) * x.shape[0]
        train_loss /= max(len(train_loader.dataset), 1)
        val_loss, y_val, p_val = run_eval(model, val_loader, criterion, device)
        p_val_for_threshold = smooth_probabilities_by_video(p_val, val_meta, args.prob_smooth_span)
        thr, val_f1 = tune_threshold(y_val, p_val_for_threshold, metric_name=args.threshold_metric)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_f1_macro": val_f1,
                "threshold": thr,
            }
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    val_loss, y_val, p_val = run_eval(model, val_loader, criterion, device)
    p_val_eval = smooth_probabilities_by_video(p_val, val_meta, args.prob_smooth_span)
    threshold, val_f1 = tune_threshold(y_val, p_val_eval, metric_name=args.threshold_metric)
    test_loss, y_test, p_test = run_eval(model, test_loader, criterion, device)
    p_test_eval = smooth_probabilities_by_video(p_test, test_meta, args.prob_smooth_span)
    metrics = binary_metrics(y_test, p_test_eval, threshold)
    metrics.update(
        {
            "name": variant["name"],
            "model": variant["model"],
            "target": variant["target"],
            "window_size": window_size,
            "window_seconds": window_size / 15.0,
            "stride": int(variant["stride"]),
            "mask": variant.get("mask", "full"),
            "label_mode": args.label_mode,
            "train_windows": len(train_samples),
            "train_windows_after_oversample": len(train_samples_for_loader),
            "val_windows": len(val_samples),
            "test_windows": len(test_samples),
            "val_loss": val_loss,
            "test_loss": test_loss,
            "val_f1_macro_tuned": val_f1,
            "epochs": args.epochs,
            "split_seed": args.seed,
            "model_seed": args.model_seed if args.model_seed is not None else args.seed,
            "seconds": time.time() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "input_size": input_size,
            "feature_standardized": not args.no_standardize,
            "positive_oversample": int(args.positive_oversample),
            "hard_negative_oversample": int(args.hard_negative_oversample),
            "hard_negative_top_frac": float(args.hard_negative_top_frac),
            "threshold_metric": args.threshold_metric,
            "label_smoothing": float(args.label_smoothing),
            "prob_smooth_span": int(args.prob_smooth_span),
            "feature_scale_std": float(args.feature_scale_std),
            "time_jitter_prob": float(args.time_jitter_prob),
            "time_jitter_max": int(args.time_jitter_max),
            "temporal_smooth_prob": float(args.temporal_smooth_prob),
        }
    )
    _train_loss, y_train_eval, p_train = run_eval(model, train_eval_loader, criterion, device)
    p_train_eval = smooth_probabilities_by_video(p_train, train_meta, args.prob_smooth_span)
    prediction_rows = []
    for split_name, meta, y_split, p_split in (
        ("train", train_meta, y_train_eval, p_train_eval),
        ("val", val_meta, y_val, p_val_eval),
        ("test", test_meta, y_test, p_test_eval),
    ):
        split_rows = build_prediction_rows(variant, meta, y_split, p_split, threshold)
        for row in split_rows:
            row["split"] = split_name
        prediction_rows.extend(split_rows)
    return metrics, history, prediction_rows


def rule_baseline(target: str, variant: dict, test_samples, label_mode: str) -> dict:
    idx = TARGETS[target]["rule_feature_index"]
    y_true = []
    y_prob = []
    for features, label in test_samples:
        y_true.append(int(label))
        vals = (features[:, idx] > 0.5).astype(np.int64)
        if label_mode == "any_positive":
            pred = int(np.any(vals == 1))
        elif label_mode == "all_positive":
            pred = int(np.all(vals == 1))
        else:
            pred = int(np.bincount(vals, minlength=2).argmax())
        y_prob.append(float(pred))
    metrics = binary_metrics(y_true, y_prob, 0.50)
    metrics.update(
        {
            "name": f"RuleBaseline-W{variant['window_size']}",
            "model": "rule_baseline",
            "target": target,
            "window_size": int(variant["window_size"]),
            "window_seconds": int(variant["window_size"]) / 15.0,
            "stride": int(variant["stride"]),
            "mask": "rule_signal",
            "label_mode": label_mode,
            "train_windows": "",
            "val_windows": "",
            "test_windows": len(test_samples),
            "val_loss": "",
            "test_loss": "",
            "val_f1_macro_tuned": "",
            "epochs": 0,
            "seconds": 0.0,
            "parameters": 0,
        }
    )
    return metrics


def write_outputs(out_dir: Path, results: List[dict], histories: Dict[str, list], config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stagewise_temporal_ablation.json").write_text(
        json.dumps({"config": config, "results": results, "histories": histories}, indent=2),
        encoding="utf-8",
    )
    fields = [
        "target",
        "name",
        "model",
        "window_size",
        "window_seconds",
        "stride",
        "mask",
        "label_mode",
        "epochs",
        "parameters",
        "train_windows",
        "val_windows",
        "test_windows",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_positive",
        "f1_macro",
        "tp",
        "tn",
        "fp",
        "fn",
        "positive_oversample",
        "hard_negative_oversample",
        "hard_negative_top_frac",
        "label_smoothing",
        "prob_smooth_span",
        "feature_scale_std",
        "time_jitter_prob",
        "time_jitter_max",
        "temporal_smooth_prob",
        "seconds",
    ]
    with (out_dir / "stagewise_temporal_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fields})

    write_markdown_summary(out_dir, results, config)
    write_plots(out_dir, results)


def write_prediction_rows(out_dir: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = [
        "split",
        "target",
        "model",
        "video",
        "video_path",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "window_size",
        "window_seconds",
        "stride",
        "known_ratio",
        "gt_label",
        "pred_label",
        "probability",
        "threshold",
        "result",
        "corrected_label",
        "review_notes",
    ]
    with (out_dir / "test_window_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def pct(value: float) -> str:
    return f"{float(value) * 100:.2f}%"


def write_markdown_summary(out_dir: Path, results: List[dict], config: dict) -> None:
    lines = [
        "# Stagewise Learned Temporal Ablation",
        "",
        "Eksperimen ini membandingkan rule-based stagewise signal dengan learned temporal model untuk `offscreen_gaze` dan `looking_away`.",
        "",
        f"- Features: `{config['features_dir']}`",
        f"- Segments: `{config['segments_csv']}`",
        f"- Split: per-video train/val/test seed `{config['seed']}`",
        f"- Label mode: `{config['label_mode']}`",
        f"- Minimum annotated window coverage: `{config.get('min_window_known_ratio', 0.0)}`",
        f"- Focused mask: `{config.get('focused_mask', '')}`",
        f"- Positive oversample: `{config.get('positive_oversample', 1)}`",
        f"- Hard-negative oversample: `{config.get('hard_negative_oversample', 1)}` "
        f"(top fraction `{config.get('hard_negative_top_frac', 0.25)}`)",
        "",
        "| Target | Model | Window | Feature setting | Accuracy | Precision | Recall | F1 positive | Macro-F1 | TP/TN/FP/FN |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in results:
        lines.append(
            "| {target} | {model} | {window:.1f}s | {mask} | {acc} | {prec} | {rec} | {f1p} | {f1m} | {tp}/{tn}/{fp}/{fn} |".format(
                target=row["target"],
                model=row["name"],
                window=float(row["window_seconds"]),
                mask=row["mask"],
                acc=pct(row["accuracy"]),
                prec=pct(row["precision"]),
                rec=pct(row["recall"]),
                f1p=pct(row["f1_positive"]),
                f1m=pct(row["f1_macro"]),
                tp=row["tp"],
                tn=row["tn"],
                fp=row["fp"],
                fn=row["fn"],
            )
        )
    (out_dir / "stagewise_temporal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(out_dir: Path, results: List[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable: {exc}")
        return

    for target in TARGETS:
        target_rows = [r for r in results if r["target"] == target]

        window_rows = [
            r
            for r in target_rows
            if r["model"] == "lstm" and r["mask"] == "full" and r["name"].startswith("LSTM-W")
        ]
        window_rows = sorted(window_rows, key=lambda r: int(r["window_size"]))
        if window_rows:
            xs = [float(r["window_seconds"]) for r in window_rows]
            ys = [float(r["f1_macro"]) * 100 for r in window_rows]
            plt.figure(figsize=(6.0, 3.5))
            plt.plot(xs, ys, marker="o", linewidth=2)
            for x, y in zip(xs, ys):
                plt.text(x, y + 0.4, f"{y:.1f}", ha="center", fontsize=9)
            plt.xlabel("Temporal window (seconds)")
            plt.ylabel("Macro-F1 (%)")
            plt.title(f"Window-size ablation: {target}")
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(out_dir / f"{target}_window_ablation.png", dpi=220)
            plt.close()

        feat_names = ["LSTM-W30-Full", "LSTM-W30-NoGaze", "LSTM-W30-NoHeadPose", "LSTM-W30-NoEyeOpen"]
        feat_rows = [next((r for r in target_rows if r["name"] == n), None) for n in feat_names]
        feat_rows = [r for r in feat_rows if r is not None]
        if feat_rows:
            labels = [r["mask"].replace("_", " ") for r in feat_rows]
            vals = [float(r["f1_macro"]) * 100 for r in feat_rows]
            plt.figure(figsize=(6.2, 3.5))
            plt.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"][: len(vals)])
            for i, v in enumerate(vals):
                plt.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
            plt.ylabel("Macro-F1 (%)")
            plt.title(f"Feature ablation: {target}")
            plt.ylim(0, max(vals + [1]) + 8)
            plt.tight_layout()
            plt.savefig(out_dir / f"{target}_feature_ablation.png", dpi=220)
            plt.close()

        compare_names = ["RuleBaseline-W30", "LSTM-W30-Full", "Transformer-W30-Full"]
        cmp_rows = [next((r for r in target_rows if r["name"] == n), None) for n in compare_names]
        cmp_rows = [r for r in cmp_rows if r is not None]
        if cmp_rows:
            labels = [r["name"].replace("-Full", "") for r in cmp_rows]
            vals = [float(r["f1_macro"]) * 100 for r in cmp_rows]
            plt.figure(figsize=(6.0, 3.4))
            plt.bar(labels, vals, color=["#7f7f7f", "#1f77b4", "#9467bd"][: len(vals)])
            for i, v in enumerate(vals):
                plt.text(i, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
            plt.ylabel("Macro-F1 (%)")
            plt.title(f"Rule vs learned temporal: {target}")
            plt.ylim(0, max(vals + [1]) + 8)
            plt.tight_layout()
            plt.savefig(out_dir / f"{target}_rule_vs_learned.png", dpi=220)
            plt.close()


def parse_window_specs(text: str | None) -> List[Tuple[int, int]] | None:
    if not text:
        return None
    specs: List[Tuple[int, int]] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        window = int(part)
        stride = max(1, window // 2)
        specs.append((window, stride))
    return specs or None


def mask_label(mask: str) -> str:
    return "".join(part.capitalize() for part in str(mask).split("_"))


def build_variants(
    targets: Sequence[str],
    mode: str,
    focused_windows: str | None = None,
    focused_mask: str = "no_target_rule",
) -> List[dict]:
    variants: List[dict] = []
    if mode == "forward":
        window_specs = parse_window_specs(focused_windows) or [(15, 8), (30, 15), (45, 23), (60, 30), (90, 45)]
        for target in targets:
            for mask, label, _ in FORWARD_FEATURE_STEPS[target]:
                for window, stride in window_specs:
                    variants.append(
                        {
                            "target": target,
                            "name": f"Forward-{label}-W{window}",
                            "model": "lstm",
                            "window_size": window,
                            "stride": stride,
                            "mask": mask,
                        }
                    )
        return variants
    if mode in {"focused", "focused_temporal", "focused_stats", "focused_transformer"}:
        window_specs = parse_window_specs(focused_windows) or [(15, 8), (30, 15), (45, 23), (60, 30), (90, 45)]
        if mode == "focused_temporal":
            model = "temporal_conv"
            label = "LearnedTemporal"
        elif mode == "focused_stats":
            model = "lstm_stats"
            label = "LSTMStats"
        elif mode == "focused_transformer":
            model = "transformer"
            label = "Transformer"
        else:
            model = "lstm"
            label = "LSTM"
        for target in targets:
            for window, stride in window_specs:
                variants.append(
                    {
                        "target": target,
                        "name": f"{label}-W{window}-{mask_label(focused_mask)}",
                        "model": model,
                        "window_size": window,
                        "stride": stride,
                        "mask": focused_mask,
                    }
                )
        return variants
    elif mode == "quick":
        window_specs = [(30, 15), (60, 30)]
        feature_specs = [
            ("full", "Full"),
            ("no_target_rule", "NoTargetRule"),
            ("no_stagewise_rules", "NoStagewiseRules"),
            ("continuous_only", "ContinuousOnly"),
            ("no_gaze", "NoGaze"),
            ("no_head_pose", "NoHeadPose"),
        ]
        include_transformer = False
    else:
        window_specs = [(15, 8), (30, 15), (45, 23), (60, 30), (90, 45)]
        feature_specs = [
            ("full", "Full"),
            ("no_target_rule", "NoTargetRule"),
            ("no_stagewise_rules", "NoStagewiseRules"),
            ("continuous_only", "ContinuousOnly"),
            ("no_gaze", "NoGaze"),
            ("no_head_pose", "NoHeadPose"),
            ("no_eye_open", "NoEyeOpen"),
            ("gaze_only", "GazeOnly"),
            ("head_pose_only", "HeadPoseOnly"),
        ]
        include_transformer = True

    for target in targets:
        for window, stride in window_specs:
            if mode == "focused":
                variants.append(
                    {
                        "target": target,
                        "name": f"LSTM-W{window}-NoTargetRule",
                        "model": "lstm",
                        "window_size": window,
                        "stride": stride,
                        "mask": "no_target_rule",
                    }
                )
            else:
                variants.append(
                    {
                        "target": target,
                        "name": f"LSTM-W{window}-Full",
                        "model": "lstm",
                        "window_size": window,
                        "stride": stride,
                        "mask": "full",
                    }
                )
        for mask, label in feature_specs:
            if mask == "full":
                continue
            variants.append(
                {
                    "target": target,
                    "name": f"LSTM-W30-{label}",
                    "model": "lstm",
                    "window_size": 30,
                    "stride": 15,
                    "mask": mask,
                }
            )
        if include_transformer:
            variants.append(
                {
                    "target": target,
                    "name": "Transformer-W30-Full",
                    "model": "transformer",
                    "window_size": 30,
                    "stride": 15,
                    "mask": "full",
                }
            )
    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/processed_final_noyolo/features")
    parser.add_argument("--segments-csv", default="data/annotations/stage_eval_segment_template.csv")
    parser.add_argument("--output-dir", default="output/reports/q1_stagewise_temporal_ablation_20260608")
    parser.add_argument("--targets", nargs="+", default=["offscreen_gaze", "looking_away"], choices=sorted(TARGETS))
    parser.add_argument(
        "--variant-mode",
        choices=["full", "quick", "forward", "focused", "focused_temporal", "focused_stats", "focused_transformer"],
        default="full",
    )
    parser.add_argument("--focused-windows", default="",
                        help="Comma-separated windows for focused modes, e.g. 90,150")
    parser.add_argument(
        "--focused-mask",
        default="no_target_rule",
        choices=["no_target_rule"] + sorted(FEATURE_GROUPS),
        help="Feature mask for focused modes. Use continuous_only for strict learned stagewise experiments.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--transformer-dim", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=None,
                        help="Model initialization seed. Defaults to --seed; split still uses --seed.")
    parser.add_argument("--split-manifest", default="",
                        help="Optional JSON with explicit train_videos/val_videos/test_videos lists.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--label-mode", choices=["majority", "any_positive", "all_positive", "positive_ratio"], default="majority")
    parser.add_argument("--positive-ratio", type=float, default=0.30)
    parser.add_argument(
        "--min-window-known-ratio",
        type=float,
        default=0.0,
        help="Minimum fraction of a temporal window covered by segment annotation before it is used.",
    )
    parser.add_argument("--threshold-metric", choices=["f1_macro", "f1_positive", "recall", "precision"], default="f1_macro")
    parser.add_argument("--positive-oversample", type=int, default=1)
    parser.add_argument("--hard-negative-oversample", type=int, default=1)
    parser.add_argument("--hard-negative-top-frac", type=float, default=0.25)
    parser.add_argument("--time-mask-prob", type=float, default=0.10)
    parser.add_argument("--feature-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-scale-std", type=float, default=0.0)
    parser.add_argument("--time-jitter-prob", type=float, default=0.0)
    parser.add_argument("--time-jitter-max", type=int, default=0)
    parser.add_argument("--temporal-smooth-prob", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--prob-smooth-span", type=int, default=1,
                        help="Odd/effective span of moving-average smoothing over per-video window probabilities.")
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    features_dir = Path(args.features_dir)
    segments_path = Path(args.segments_csv)
    if args.split_manifest:
        split_data = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        train_videos = [str(v) for v in split_data.get("train_videos", [])]
        val_videos = [str(v) for v in split_data.get("val_videos", [])]
        test_videos = [str(v) for v in split_data.get("test_videos", [])]
    else:
        train_videos, val_videos, test_videos = split_videos(
            str(features_dir),
            str(features_dir),  # split_videos only needs matching names; features dir works for both here.
            args.train_ratio,
            args.val_ratio,
            args.seed,
        )

    # split_videos expects both dirs to contain same filenames. Passing features_dir twice
    # keeps the same deterministic per-video split without relying on global labels.
    variants = build_variants(
        args.targets,
        args.variant_mode,
        focused_windows=args.focused_windows,
        focused_mask=args.focused_mask,
    )
    results: List[dict] = []
    histories: Dict[str, list] = {}
    prediction_rows: List[dict] = []
    config = {
        "features_dir": str(features_dir),
        "segments_csv": str(segments_path),
        "output_dir": str(out_dir),
        "split_manifest": args.split_manifest,
        "targets": args.targets,
        "variant_mode": args.variant_mode,
        "epochs": args.epochs,
        "seed": args.seed,
        "model_seed": args.model_seed if args.model_seed is not None else args.seed,
        "label_mode": args.label_mode,
        "positive_ratio": float(args.positive_ratio),
        "min_window_known_ratio": float(args.min_window_known_ratio),
        "threshold_metric": args.threshold_metric,
        "positive_oversample": int(args.positive_oversample),
        "hard_negative_oversample": int(args.hard_negative_oversample),
        "hard_negative_top_frac": float(args.hard_negative_top_frac),
        "label_smoothing": float(args.label_smoothing),
        "prob_smooth_span": int(args.prob_smooth_span),
        "feature_scale_std": float(args.feature_scale_std),
        "time_jitter_prob": float(args.time_jitter_prob),
        "time_jitter_max": int(args.time_jitter_max),
        "temporal_smooth_prob": float(args.temporal_smooth_prob),
        "focused_mask": args.focused_mask,
        "feature_standardized": not args.no_standardize,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
        "legacy_name_to_canonical": LEGACY_NAME_TO_CANONICAL,
        "special_video_aliases": SPECIAL_VIDEO_ALIASES,
    }

    label_cache: Dict[str, Dict[str, np.ndarray]] = {}
    known_cache: Dict[str, Dict[str, np.ndarray]] = {}
    for target in args.targets:
        labels, known, mapping_rows = build_target_labels(segments_path, features_dir, target, out_dir)
        label_cache[target] = labels
        known_cache[target] = known
        print(f"[INFO] target={target} labelled_videos={len(labels)} mapping_rows={len(mapping_rows)}")

    for variant in variants:
        target = variant["target"]
        labels = label_cache[target]
        known = known_cache[target]
        train_samples, _train_meta = make_samples(
            features_dir,
            labels,
            known,
            train_videos,
            int(variant["window_size"]),
            int(variant["stride"]),
            args.label_mode,
            positive_ratio=args.positive_ratio,
            min_known_ratio=args.min_window_known_ratio,
        )
        val_samples, val_meta = make_samples(
            features_dir,
            labels,
            known,
            val_videos,
            int(variant["window_size"]),
            int(variant["stride"]),
            args.label_mode,
            positive_ratio=args.positive_ratio,
            min_known_ratio=args.min_window_known_ratio,
        )
        test_samples, test_meta = make_samples(
            features_dir,
            labels,
            known,
            test_videos,
            int(variant["window_size"]),
            int(variant["stride"]),
            args.label_mode,
            positive_ratio=args.positive_ratio,
            min_known_ratio=args.min_window_known_ratio,
        )
        if not train_samples or not val_samples or not test_samples:
            print(f"[WARN] skipping {variant['name']} target={target}: empty split")
            continue

        if variant["model"] == "lstm" and variant["mask"] == "full":
            baseline = rule_baseline(target, variant, test_samples, args.label_mode)
            results.append(baseline)

        print(
            f"\n=== {target} {variant['name']} "
            f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} ==="
        )
        metrics, history, predictions = train_model(
            args,
            variant,
            train_samples,
            val_samples,
            test_samples,
            _train_meta,
            val_meta,
            test_meta,
        )
        results.append(metrics)
        histories[f"{target}:{variant['name']}"] = history
        prediction_rows.extend(predictions)
        print(
            json.dumps(
                {
                    "target": metrics["target"],
                    "name": metrics["name"],
                    "accuracy": round(metrics["accuracy"] * 100, 2),
                    "precision": round(metrics["precision"] * 100, 2),
                    "recall": round(metrics["recall"] * 100, 2),
                    "f1_positive": round(metrics["f1_positive"] * 100, 2),
                    "f1_macro": round(metrics["f1_macro"] * 100, 2),
                    "threshold": round(metrics["threshold"], 2),
                },
                indent=2,
            )
        )
        write_outputs(out_dir, results, histories, config)

    write_outputs(out_dir, results, histories, config)
    write_prediction_rows(out_dir, prediction_rows)
    print(f"\n[OK] Saved stagewise temporal ablation to {out_dir}")


if __name__ == "__main__":
    main()
