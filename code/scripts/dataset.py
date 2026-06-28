"""
PyTorch Dataset & DataLoader untuk Temporal Cheating Detection
==============================================================
Memuat feature sequences + labels untuk training LSTM model.

Mendukung:
- Load dari .npy files (feature arrays per video)
- Sliding window dengan configurable overlap
- Train/val/test split
- Data augmentation: Gaussian noise, time stretch
"""

import os
import re
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Dataset module requires PyTorch.")


# ============================================================================
# Label definitions (konsisten dengan temporal_model.py)
# ============================================================================
LABEL_MAP = {
    "normal": 0,
    "suspicious": 1,       # Any cheating indicator
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


# ============================================================================
# Dataset Class
# ============================================================================
if TORCH_AVAILABLE:

    class CheatingDetectionDataset(Dataset):
        """
        Dataset untuk deteksi kecurangan temporal.

        Setiap sample adalah:
        - Input: sequence of feature vectors (window_size, num_features)
        - Label: kelas biner (0=normal, 1=suspicious)

        Data structure di disk:
        data/processed/
        ├── features/
        │   ├── video_001.npy    # (num_frames, num_features)
        │   ├── video_002.npy
        │   └── ...
        └── labels/
            ├── video_001.npy    # (num_frames,) integer labels
            ├── video_002.npy
            └── ...
        """

        def __init__(self,
                     features_dir: str = "data/processed/features",
                     labels_dir: str = "data/processed/labels",
                     window_size: int = 30,
                     stride: int = 15,
                     augment: bool = False,
                     noise_std: float = 0.01,
                     augment_mask_prob: float = 0.30,
                     augment_mask_max_frac: float = 0.25,
                     augment_feature_dropout_prob: float = 0.0,
                     augment_time_reverse_prob: float = 0.0,
                     video_list: Optional[List[str]] = None,
                     label_mode: str = "majority",
                     normalize_per_video: bool = False):
            """
            Args:
                features_dir: Folder berisi .npy feature files per video
                labels_dir: Folder berisi .npy label files per video
                window_size: Jumlah frame per sample
                stride: Step size antara windows (overlap = window_size - stride)
                augment: Apakah augmentasi aktif
                noise_std: Standar deviasi Gaussian noise untuk augmentasi
                video_list: Daftar nama video untuk digunakan (tanpa .npy)
                           None = gunakan semua yang tersedia
            """
            self.window_size = window_size
            self.stride = stride
            self.augment = augment
            self.noise_std = noise_std
            self.augment_mask_prob = float(max(0.0, min(1.0, augment_mask_prob)))
            self.augment_mask_max_frac = float(max(0.05, min(0.90, augment_mask_max_frac)))
            self.augment_feature_dropout_prob = float(max(0.0, min(1.0, augment_feature_dropout_prob)))
            self.augment_time_reverse_prob = float(max(0.0, min(1.0, augment_time_reverse_prob)))
            self.normalize_per_video = bool(normalize_per_video)
            self.label_mode = label_mode
            if self.label_mode not in ("majority", "any_suspicious", "all_suspicious"):
                raise ValueError(
                    f"Unsupported label_mode: {self.label_mode}. "
                    "Use one of: majority, any_suspicious, all_suspicious"
                )

            self.features_dir = Path(features_dir)
            self.labels_dir = Path(labels_dir)

            # Kumpulkan semua video files
            if video_list is not None:
                video_files = [f"{v}.npy" for v in video_list]
            else:
                video_files = sorted([
                    f.name for f in self.features_dir.glob("*.npy")
                    if (self.labels_dir / f.name).exists()
                ])

            # Load semua data dan buat windows
            self.samples = []
            self._load_data(video_files)

            norm_tag = ",normalized" if self.normalize_per_video else ""
            print(f"[Dataset] Loaded {len(self.samples)} windows "
                  f"from {len(video_files)} videos "
                  f"(label_mode={self.label_mode}{norm_tag}) "
                  f"(window={window_size}, stride={stride})")

        def _window_label(self, window_labels: np.ndarray) -> int:
            """Convert per-frame labels in a window into one binary window label."""
            binary_labels = (window_labels.astype(int) > 0).astype(np.int64)
            if self.label_mode == "any_suspicious":
                return int(np.any(binary_labels == 1))
            if self.label_mode == "all_suspicious":
                return int(np.all(binary_labels == 1))
            # default: majority
            return int(np.bincount(binary_labels).argmax())

        def _load_data(self, video_files: List[str]):
            """Load semua video features/labels dan buat sliding windows."""
            for vf in video_files:
                feat_path = self.features_dir / vf
                label_path = self.labels_dir / vf

                if not feat_path.exists() or not label_path.exists():
                    print(f"[WARNING] Skipping {vf}: file not found")
                    continue

                features = np.load(str(feat_path))      # (T, F)
                labels = np.load(str(label_path))        # (T,)

                if len(features) != len(labels):
                    print(f"[WARNING] Skipping {vf}: feature/label length mismatch")
                    continue

                # Per-video z-score normalization.
                # Menghilangkan bias nilai absolut antar subjek (posisi kamera, postur, dll)
                # agar model belajar dari deviasi relatif masing-masing subjek.
                if self.normalize_per_video and len(features) > 1:
                    mean = features.mean(axis=0, keepdims=True)  # (1, F)
                    std = features.std(axis=0, keepdims=True)    # (1, F)
                    features = (features - mean) / (std + 1e-8)

                # Sliding window
                num_frames = len(features)
                for start in range(0, num_frames - self.window_size + 1, self.stride):
                    end = start + self.window_size
                    window_features = features[start:end]
                    window_labels = labels[start:end]

                    window_label = self._window_label(window_labels)

                    self.samples.append({
                        "features": window_features.astype(np.float32),
                        "label": window_label,
                        "video": vf,
                        "start_frame": start,
                        "end_frame": end,
                    })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = self.samples[idx]
            features = torch.tensor(sample["features"], dtype=torch.float32)
            label = torch.tensor(sample["label"], dtype=torch.long)

            # Augmentasi
            if self.augment:
                features = self._augment(features)

            return features, label

        def _augment(self, features: torch.Tensor) -> torch.Tensor:
            """Apply data augmentation."""
            # Gaussian noise
            if self.noise_std > 0:
                noise = torch.randn_like(features) * self.noise_std
                features = features + noise

            # Random time masking (mask beberapa frame)
            if np.random.random() < self.augment_mask_prob:
                max_len = max(1, int(self.window_size * self.augment_mask_max_frac))
                mask_len = np.random.randint(1, max_len + 1)
                if self.window_size - mask_len > 0:
                    mask_start = np.random.randint(0, self.window_size - mask_len + 1)
                    features[mask_start:mask_start + mask_len] = 0.0

            # Feature dropout across the whole sequence (channel-wise)
            if self.augment_feature_dropout_prob > 0:
                fdim = int(features.shape[1])
                keep = (torch.rand(fdim, device=features.device) > self.augment_feature_dropout_prob).float()
                features = features * keep.unsqueeze(0)

            # Temporal sequence reversal
            # Pola perilaku mencurigakan umumnya tetap terdeteksi meski urutan dibalik
            # (misal: window dengan banyak frame looking_away tetap mencurigakan di-reverse)
            if self.augment_time_reverse_prob > 0 and np.random.random() < self.augment_time_reverse_prob:
                features = torch.flip(features, dims=[0])

            return features

        def get_class_weights(self) -> torch.Tensor:
            """Hitung class weights untuk mengatasi class imbalance."""
            labels = [s["label"] for s in self.samples]
            counts = np.bincount(labels, minlength=len(LABEL_MAP))
            total = len(labels)

            weights = total / (len(LABEL_MAP) * counts + 1e-8)
            weights = weights / weights.sum() * len(LABEL_MAP)

            return torch.tensor(weights, dtype=torch.float32)

        def get_label_distribution(self) -> Dict[str, int]:
            """Distribusi label dalam dataset."""
            labels = [s["label"] for s in self.samples]
            counts = np.bincount(labels, minlength=len(LABEL_MAP))
            return {LABEL_NAMES[i]: int(c) for i, c in enumerate(counts)}


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_subject_id(video_stem: str) -> str:
    """Extract participant id such as p01 from a cached video stem."""
    m = re.search(r"(p\d+)_vid_", str(video_stem).lower())
    if not m:
        raise ValueError(f"Cannot extract subject id from video stem: {video_stem}")
    return m.group(1)


def split_videos(
    features_dir: str = "data/processed/features",
    labels_dir: str = "data/processed/labels",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    split_unit: str = "video",
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split video names into train/val/test.

    split_unit="video" keeps the legacy per-video split.
    split_unit="subject" groups all videos from the same participant in the same split.
    Returns video stems without .npy extension.
    """
    features_path = Path(features_dir)
    labels_path = Path(labels_dir)

    all_videos = sorted([
        f.stem for f in features_path.glob("*.npy")
        if (labels_path / f.name).exists()
    ])

    if len(all_videos) == 0:
        raise ValueError(f"No matching feature/label files found in "
                         f"{features_dir} and {labels_dir}")

    split_unit = str(split_unit).lower().strip()
    np.random.seed(seed)
    if split_unit == "subject":
        subject_to_videos: Dict[str, List[str]] = {}
        for video in all_videos:
            subject_to_videos.setdefault(_extract_subject_id(video), []).append(video)

        subjects = sorted(subject_to_videos)
        np.random.shuffle(subjects)

        n = len(subjects)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        train_subjects = subjects[:n_train]
        val_subjects = subjects[n_train:n_train + n_val]
        test_subjects = subjects[n_train + n_val:]

        train_videos = sorted(v for s in train_subjects for v in subject_to_videos[s])
        val_videos = sorted(v for s in val_subjects for v in subject_to_videos[s])
        test_videos = sorted(v for s in test_subjects for v in subject_to_videos[s])
        return train_videos, val_videos, test_videos

    if split_unit != "video":
        raise ValueError(f"Unsupported split_unit: {split_unit}. Use 'video' or 'subject'.")

    np.random.shuffle(all_videos)
    n = len(all_videos)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))

    train_videos = all_videos[:n_train]
    val_videos = all_videos[n_train:n_train + n_val]
    test_videos = all_videos[n_train + n_val:]
    return train_videos, val_videos, test_videos


def create_dataloaders(
    features_dir: str = "data/processed/features",
    labels_dir: str = "data/processed/labels",
    window_size: int = 30,
    stride: int = 15,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    label_mode: str = "majority",
    augment_noise_std: float = 0.01,
    augment_mask_prob: float = 0.30,
    augment_mask_max_frac: float = 0.25,
    augment_feature_dropout_prob: float = 0.0,
    augment_time_reverse_prob: float = 0.0,
    video_splits: Optional[Tuple[List[str], List[str], List[str]]] = None,
    split_unit: str = "video",
    normalize_per_video: bool = False,
) -> Tuple:
    """
    Buat train/val/test DataLoaders.

    Split dilakukan per-video (bukan per-window) untuk mencegah data leakage.

    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")

    if video_splits is None:
        train_videos, val_videos, test_videos = split_videos(
            features_dir=features_dir,
            labels_dir=labels_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            split_unit=split_unit,
        )
    else:
        train_videos, val_videos, test_videos = video_splits

    print(f"[Split] Train: {len(train_videos)} videos, "
          f"Val: {len(val_videos)} videos, "
          f"Test: {len(test_videos)} videos")

    # Buat datasets
    train_ds = CheatingDetectionDataset(
        features_dir, labels_dir, window_size, stride,
        augment=True,
        noise_std=augment_noise_std,
        augment_mask_prob=augment_mask_prob,
        augment_mask_max_frac=augment_mask_max_frac,
        augment_feature_dropout_prob=augment_feature_dropout_prob,
        augment_time_reverse_prob=augment_time_reverse_prob,
        video_list=train_videos,
        label_mode=label_mode,
        normalize_per_video=normalize_per_video,
    )
    val_ds = CheatingDetectionDataset(
        features_dir, labels_dir, window_size, stride,
        augment=False, video_list=val_videos, label_mode=label_mode,
        normalize_per_video=normalize_per_video,
    )
    test_ds = CheatingDetectionDataset(
        features_dir, labels_dir, window_size, stride,
        augment=False, video_list=test_videos, label_mode=label_mode,
        normalize_per_video=normalize_per_video,
    )

    # Buat dataloaders
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    class_weights = train_ds.get_class_weights()

    # Print distribusi
    print(f"\n[Train] Distribution: {train_ds.get_label_distribution()}")
    print(f"[Val]   Distribution: {val_ds.get_label_distribution()}")
    print(f"[Test]  Distribution: {test_ds.get_label_distribution()}")
    print(f"[Class Weights] {class_weights.tolist()}")

    return train_loader, val_loader, test_loader, class_weights


# ============================================================================
# Quick Test
# ============================================================================
if __name__ == "__main__":
    print("=== Dataset Module Test ===")
    print(f"TORCH_AVAILABLE: {TORCH_AVAILABLE}")
    print(f"LABEL_MAP: {LABEL_MAP}")

    # Test: buat dummy data
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        feat_dir = os.path.join(tmpdir, "features")
        label_dir = os.path.join(tmpdir, "labels")
        os.makedirs(feat_dir)
        os.makedirs(label_dir)

        # Buat 3 dummy videos
        for i in range(3):
            n_frames = np.random.randint(60, 120)
            n_features = 15
            features = np.random.randn(n_frames, n_features).astype(np.float32)
            labels = np.random.randint(0, 2, size=(n_frames,)).astype(np.int64)

            np.save(os.path.join(feat_dir, f"video_{i:03d}.npy"), features)
            np.save(os.path.join(label_dir, f"video_{i:03d}.npy"), labels)

        if TORCH_AVAILABLE:
            train_dl, val_dl, test_dl, weights = create_dataloaders(
                feat_dir, label_dir,
                window_size=30, stride=10, batch_size=4,
                train_ratio=0.6, val_ratio=0.2
            )

            for batch_feat, batch_label in train_dl:
                print(f"\nBatch features shape: {batch_feat.shape}")
                print(f"Batch labels shape: {batch_label.shape}")
                print(f"Batch labels: {batch_label.tolist()}")
                break

    print("\n[OK] Dataset module test passed!")
