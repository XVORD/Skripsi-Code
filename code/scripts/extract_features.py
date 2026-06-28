"""
Feature Extraction dari Video
==============================
Menjalankan pipeline pada semua video di dataset,
mengekstrak feature vectors per frame, dan menyimpan sebagai .npy files.

Usage:
    python scripts/extract_features.py --input data/raw_videos --output data/processed
    python scripts/extract_features.py --input data/raw_videos/normal --output data/processed --label normal
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.face_detection import FaceDetector
from src.head_pose_estimator import HeadPoseEstimator
from src.eye_gaze_estimator import EyeGazeEstimator
from src.object_detection import ObjectDetector
from src.behavior_features import BehaviorFeatureExtractor
from src.video_acquisition import VideoAcquisition
from src.behavior_features import BehaviorFeatureVector


# Label mapping
LABEL_MAP = {
    "normal": 0,
    "third_party": 1,
    "ai_assistance": 1,
    "suspicious": 1,
}


def extract_features_from_video(
    video_path: str,
    config_path: str = "configs/config.yaml",
    use_object_detection: bool = True,
) -> np.ndarray:
    """
    Ekstrak feature vectors dari satu video.

    Args:
        video_path: Path ke file video
        config_path: Path ke config YAML
        use_object_detection: Aktifkan YOLO (lebih lambat tapi lebih lengkap)

    Returns:
        features: np.ndarray shape (num_frames, num_features)
    """
    # Init modules
    va = VideoAcquisition(config_path)
    fd = FaceDetector(config_path)
    hpe = HeadPoseEstimator(config_path)
    ege = EyeGazeEstimator(config_path)
    bfe = BehaviorFeatureExtractor(config_path)

    od = None
    if use_object_detection:
        od = ObjectDetector(config_path)
        if not od.initialize():
            print(f"[WARNING] YOLO not available, proceeding without object detection")
            od = None

    # Open video
    if not va.start(video_path):
        raise RuntimeError(f"Cannot open video: {video_path}")

    features_list = []
    fps = va.actual_fps or 15.0

    for frame_num, original_bgr, preprocessed_rgb in va.stream_frames():
        timestamp = frame_num / fps

        # Face detection
        face_result = fd.detect(preprocessed_rgb, frame_num, timestamp)

        # Head pose + eye gaze (hanya jika ada wajah)
        head_result = None
        gaze_result = None

        if face_result.face_present and len(face_result.faces) > 0:
            primary_face = face_result.faces[0]

            # Head pose
            if "pose_landmarks_2d" in primary_face:
                head_result = hpe.estimate(
                    primary_face["pose_landmarks_2d"],
                    primary_face.get("img_w", original_bgr.shape[1]),
                    primary_face.get("img_h", original_bgr.shape[0]),
                    frame_num, timestamp
                )

            # Eye gaze
            gaze_result = ege.estimate(primary_face, frame_num, timestamp, head_result)

        # Object detection
        obj_result = None
        if od is not None:
            obj_result = od.detect(original_bgr, frame_num, timestamp)

        # Extract behavior features
        feature_vec = bfe.extract(
            face_result, head_result, gaze_result,
            obj_result, frame_num, timestamp
        )

        features_list.append(feature_vec.to_array())

    # Cleanup
    va.stop()
    fd.release()

    if len(features_list) == 0:
        return np.array([])

    return np.array(features_list, dtype=np.float32)


def process_dataset(
    input_dir: str,
    output_dir: str,
    labels_file: Optional[str] = None,
    default_label: str = "normal",
    config_path: str = "configs/config.yaml",
    use_object_detection: bool = True,
):
    """
    Proses seluruh folder video dan simpan feature + label files.

    Struktur input:
        input_dir/
        ├── normal/           # Subfolder = label
        │   ├── video_001.mp4
        │   └── video_002.mp4
        ├── third_party/
        └── ai_assistance/

    ATAU jika labels_file diberikan:
        input_dir/
        ├── video_001.mp4
        └── video_002.mp4

        labels_file (JSON):
        {"video_001": "normal", "video_002": "third_party"}

    Output:
        output_dir/
        ├── features/
        │   ├── video_001.npy
        │   └── video_002.npy
        └── labels/
            ├── video_001.npy
            └── video_002.npy
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    feat_dir = output_path / "features"
    label_dir = output_path / "labels"
    feat_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Determine video files and labels
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
    videos = []  # List of (path, label_name)

    if labels_file:
        # Load labels from JSON
        with open(labels_file, "r") as f:
            label_dict = json.load(f)

        for vid_file in input_path.iterdir():
            if vid_file.suffix.lower() in video_extensions:
                label_name = label_dict.get(vid_file.stem, default_label)
                videos.append((vid_file, label_name))
    else:
        # Subfolder-based labels
        for subfolder in input_path.iterdir():
            if subfolder.is_dir() and subfolder.name in LABEL_MAP:
                for vid_file in subfolder.iterdir():
                    if vid_file.suffix.lower() in video_extensions:
                        videos.append((vid_file, subfolder.name))

        # Also check root level
        for vid_file in input_path.iterdir():
            if vid_file.suffix.lower() in video_extensions:
                videos.append((vid_file, default_label))

    if not videos:
        print(f"[ERROR] No video files found in {input_dir}")
        return

    print(f"[INFO] Found {len(videos)} video files")
    print(f"[INFO] Label distribution: ", end="")
    label_counts = {}
    for _, label in videos:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(label_counts)

    # Process each video
    success_count = 0
    for vid_path, label_name in tqdm(videos, desc="Extracting features"):
        try:
            unique_id = f"{label_name}__{vid_path.stem}"
            feature_out_path = feat_dir / f"{unique_id}.npy"
            label_out_path = label_dir / f"{unique_id}.npy"

            # Resume-friendly behavior: skip videos that are already processed.
            if feature_out_path.exists() and label_out_path.exists():
                success_count += 1
                print(f"  [SKIP] {vid_path.name}: existing outputs found ({unique_id})")
                continue

            features = extract_features_from_video(
                str(vid_path), config_path, use_object_detection
            )

            if len(features) == 0:
                print(f"  [SKIP] {vid_path.name}: no features extracted")
                continue

            # Prefer ground-truth labels (hasil convert_ground_truth.py) jika tersedia.
            # Fallback ke label statis per-video hanya jika file GT tidak ada.
            if label_out_path.exists():
                labels = np.load(str(label_out_path))
                if len(labels) != len(features):
                    min_len = min(len(labels), len(features))
                    print(
                        f"  [WARN] {vid_path.name}: feature/GT length mismatch "
                        f"({len(features)} vs {len(labels)}), truncating to {min_len}"
                    )
                    features = features[:min_len]
                    labels = labels[:min_len]
            else:
                label_id = LABEL_MAP.get(label_name, 0)
                labels = np.full(len(features), label_id, dtype=np.int64)

            # Save
            np.save(str(feature_out_path), features)
            np.save(str(label_out_path), labels)

            success_count += 1
            print(f"  [OK] {vid_path.name}: {len(features)} frames -> {unique_id}")

        except Exception as e:
            print(f"  [ERROR] {vid_path.name}: {e}")

    print(f"\n[DONE] Extracted features from {success_count}/{len(videos)} videos")
    print(f"[DONE] Output saved to {output_dir}")

    # Save metadata
    metadata = {
        "num_videos": success_count,
        "label_map": LABEL_MAP,
        "feature_names": BehaviorFeatureVector.feature_names(),
        "num_features": BehaviorFeatureVector.num_features(),
    }
    with open(str(output_path / "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract behavior features from video dataset"
    )
    parser.add_argument("--input", "-i", type=str, default="data/raw_videos",
                       help="Input directory containing videos")
    parser.add_argument("--output", "-o", type=str, default="data/processed",
                       help="Output directory for features/labels")
    parser.add_argument("--labels-file", type=str, default=None,
                       help="JSON file mapping video names to labels")
    parser.add_argument("--default-label", type=str, default="normal",
                       choices=list(LABEL_MAP.keys()),
                       help="Default label when not in subfolder/labels-file")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to config YAML")
    parser.add_argument("--no-yolo", action="store_true",
                       help="Disable YOLOv8 object detection (faster)")

    args = parser.parse_args()

    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        labels_file=args.labels_file,
        default_label=args.default_label,
        config_path=args.config,
        use_object_detection=not args.no_yolo,
    )
