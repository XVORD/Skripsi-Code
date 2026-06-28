"""
Dataset Preprocessing Script
==============================
Preprocessing video dataset: resize, normalize, split train/val/test.

Usage:
    python scripts/preprocess_dataset.py --input data/raw_videos --output data/processed
    python scripts/preprocess_dataset.py --input data/raw_videos --output data/processed --extract-frames
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import cv2
except ImportError:
    print("[ERROR] opencv-python not installed.")
    sys.exit(1)


LABEL_MAP = {
    "normal": 0,
    "third_party": 1,
    "ai_assistance": 1,
    "suspicious": 1,
}


def get_video_info(video_path: str) -> dict:
    """Dapatkan info video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    info = {
        "path": video_path,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info


def resize_video(input_path: str, output_path: str,
                 target_width: int = 640, target_height: int = 480,
                 target_fps: int = 15):
    """Resize dan normalize video."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {input_path}")
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps,
                            (target_width, target_height))

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(source_fps / target_fps))
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            resized = cv2.resize(frame, (target_width, target_height))
            writer.write(resized)

        frame_idx += 1

    cap.release()
    writer.release()
    return True


def extract_frames(video_path: str, output_dir: str,
                   target_fps: int = 15, max_frames: int = None):
    """Extract frames dari video sebagai PNG files."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    os.makedirs(output_dir, exist_ok=True)

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_skip = max(1, int(source_fps / target_fps))
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

            if max_frames and saved_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved_count


def split_dataset(video_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split video list ke train/val/test."""
    np.random.seed(seed)
    indices = np.random.permutation(len(video_list))

    n_train = max(1, int(len(video_list) * train_ratio))
    n_val = max(1, int(len(video_list) * val_ratio))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    return (
        [video_list[i] for i in train_idx],
        [video_list[i] for i in val_idx],
        [video_list[i] for i in test_idx],
    )


def preprocess_dataset(args):
    """Main preprocessing function."""
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"\n{'='*60}")
    print(f"Dataset Preprocessing")
    print(f"{'='*60}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Collect videos (by subfolder labels)
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
    all_videos = []  # (path, label)

    for subfolder in input_path.iterdir():
        if subfolder.is_dir() and subfolder.name in LABEL_MAP:
            for vid in subfolder.iterdir():
                if vid.suffix.lower() in video_extensions:
                    all_videos.append((vid, subfolder.name))

    # Also check root
    for vid in input_path.iterdir():
        if vid.suffix.lower() in video_extensions:
            all_videos.append((vid, "normal"))

    if not all_videos:
        print(f"[ERROR] No videos found in {input_path}")
        print(f"Expected structure:")
        print(f"  {input_path}/")
        print(f"  +- normal/")
        print(f"  +- third_party/")
        print(f"  +- ai_assistance/")
        print(f"  +- suspicious/")
        return

    print(f"\nFound {len(all_videos)} videos")

    # Label distribution
    label_counts = {}
    for _, label in all_videos:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"Label distribution: {label_counts}")

    # Split
    train_vids, val_vids, test_vids = split_dataset(
        all_videos, args.train_ratio, args.val_ratio, args.seed
    )
    print(f"Split: train={len(train_vids)}, val={len(val_vids)}, test={len(test_vids)}")

    # Create directory structure
    for split_name in ["train", "val", "test"]:
        (output_path / "videos" / split_name).mkdir(parents=True, exist_ok=True)
        if args.extract_frames:
            (output_path / "frames" / split_name).mkdir(parents=True, exist_ok=True)

    # Process videos
    metadata = {"train": [], "val": [], "test": []}

    for split_name, split_vids in [("train", train_vids), ("val", val_vids), ("test", test_vids)]:
        print(f"\nProcessing {split_name} set ({len(split_vids)} videos)...")
        for vid_path, label in split_vids:
            # Get info
            info = get_video_info(str(vid_path))
            info["label"] = label
            info["label_id"] = LABEL_MAP[label]

            # Resize video
            out_vid = output_path / "videos" / split_name / f"{vid_path.stem}.mp4"
            success = resize_video(
                str(vid_path), str(out_vid),
                args.target_width, args.target_height, args.target_fps
            )

            if success:
                info["processed_path"] = str(out_vid)
                metadata[split_name].append(info)
                print(f"  [OK] {vid_path.name} -> {label}")

                # Extract frames
                if args.extract_frames:
                    frame_dir = output_path / "frames" / split_name / vid_path.stem
                    n_frames = extract_frames(str(vid_path), str(frame_dir),
                                            args.target_fps, args.max_frames)
                    info["num_frames_extracted"] = n_frames
            else:
                print(f"  [ERR] {vid_path.name}")

    # Save metadata
    meta_path = output_path / "metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2,  default=str)

    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"  Videos: {output_path / 'videos'}")
    if args.extract_frames:
        print(f"  Frames: {output_path / 'frames'}")
    print(f"  Metadata: {meta_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess video dataset for proctoring system"
    )
    parser.add_argument("--input", "-i", type=str, default="data/raw_videos")
    parser.add_argument("--output", "-o", type=str, default="data/processed")
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-frames", action="store_true",
                       help="Also extract individual frames")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Max frames per video (for frame extraction)")

    args = parser.parse_args()
    preprocess_dataset(args)
