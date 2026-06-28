"""
Extract feature cache for the official train/val/test split manifest.

This wrapper keeps the output filenames identical to the split manifest
(`scenario__video_stem.npy`) and can resolve videos that were moved into
"Keep not use" folders after the original split was created.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.extract_features import LABEL_MAP, extract_features_from_video
from src.behavior_features import BehaviorFeatureVector


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}


def manifest_video_ids(manifest_path: Path) -> List[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids: List[str] = []
    for key in ("train_videos", "val_videos", "test_videos"):
        ids.extend(str(v) for v in manifest.get(key, []))
    seen = set()
    unique = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def index_videos(root: Path) -> Dict[Tuple[str, str], List[Path]]:
    index: Dict[Tuple[str, str], List[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            continue
        scenario = rel.parts[0]
        if scenario not in LABEL_MAP:
            continue
        index.setdefault((scenario, path.stem.lower()), []).append(path)
    return index


def candidate_stems(stem: str) -> List[str]:
    stems = [stem]
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].startswith("p") and parts[0][1:].isdigit():
        pnum = int(parts[0][1:])
        stems.extend(
            [
                f"p{pnum:02d}_vid_{pnum:03d}_vid_001",
                f"p{pnum:02d}_vid_{pnum:03d}_vid_002",
                f"p{pnum:02d}_vid_020_vid_002",
                f"P{pnum:02d}_vid_{pnum:03d}",
            ]
        )
    out = []
    for item in stems:
        low = item.lower()
        if low not in out:
            out.append(low)
    return out


def choose_path(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    if not paths:
        raise ValueError("empty candidate path list")
    # Prefer files outside "Keep not use" unless the manifest video only exists there.
    paths.sort(key=lambda p: (("keep not use" in str(p).lower()), len(str(p)), str(p).lower()))
    return paths[0]


def resolve_video(video_id: str, index: Dict[Tuple[str, str], List[Path]]) -> Path | None:
    scenario, stem = video_id.split("__", 1)
    candidates: List[Path] = []
    for candidate in candidate_stems(stem):
        candidates.extend(index.get((scenario, candidate), []))
    if not candidates:
        return None
    return choose_path(candidates)


def write_manifest_report(out_dir: Path, rows: List[dict]) -> None:
    import csv

    fields = ["video_id", "split", "status", "source_path", "frames", "message"]
    with (out_dir / "feature_manifest_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def split_lookup(manifest_path: Path) -> Dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lookup = {}
    for split in ("train", "val", "test"):
        for video_id in manifest.get(f"{split}_videos", []):
            lookup[str(video_id)] = split
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="models/split_manifest.json")
    parser.add_argument("--videos-root", default="data/videos")
    parser.add_argument("--output", default="data/processed_stagewise_latest_noyolo_20260609")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    videos_root = Path(args.videos_root)
    output_dir = Path(args.output)
    feat_dir = output_dir / "features"
    label_dir = output_dir / "labels"
    feat_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    video_ids = manifest_video_ids(manifest_path)
    splits = split_lookup(manifest_path)
    video_index = index_videos(videos_root)

    rows: List[dict] = []
    resolved: List[Tuple[str, Path]] = []
    for video_id in video_ids:
        path = resolve_video(video_id, video_index)
        if path is None:
            rows.append(
                {
                    "video_id": video_id,
                    "split": splits.get(video_id, ""),
                    "status": "missing_source",
                    "message": "No matching video file found",
                }
            )
            continue
        resolved.append((video_id, path))
        rows.append(
            {
                "video_id": video_id,
                "split": splits.get(video_id, ""),
                "status": "resolved",
                "source_path": str(path),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_report(output_dir, rows)

    missing = [r for r in rows if r["status"] == "missing_source"]
    print(f"[INFO] Manifest videos: {len(video_ids)}")
    print(f"[INFO] Resolved videos: {len(resolved)}")
    print(f"[INFO] Missing videos: {len(missing)}")
    if missing:
        for row in missing:
            print(f"[MISSING] {row['video_id']}")
    if args.dry_run:
        print(f"[OK] Dry-run report: {output_dir / 'feature_manifest_report.csv'}")
        return

    success = 0
    for video_id, source_path in tqdm(resolved, desc="Extracting manifest features"):
        scenario = video_id.split("__", 1)[0]
        feature_path = feat_dir / f"{video_id}.npy"
        label_path = label_dir / f"{video_id}.npy"
        if feature_path.exists() and label_path.exists() and not args.force:
            success += 1
            continue
        try:
            features = extract_features_from_video(
                str(source_path),
                config_path=args.config,
                use_object_detection=not args.no_yolo,
            )
            if len(features) == 0:
                print(f"[SKIP] {video_id}: no features extracted")
                continue
            np.save(str(feature_path), features.astype(np.float32))
            labels = np.full(len(features), LABEL_MAP.get(scenario, 0), dtype=np.int64)
            np.save(str(label_path), labels)
            success += 1
            print(f"[OK] {video_id}: {len(features)} frames <- {source_path}")
        except Exception as exc:
            print(f"[ERROR] {video_id}: {exc}")

    metadata = {
        "source_manifest": str(manifest_path),
        "videos_root": str(videos_root),
        "num_manifest_videos": len(video_ids),
        "num_resolved_videos": len(resolved),
        "num_success": success,
        "label_map": LABEL_MAP,
        "feature_names": BehaviorFeatureVector.feature_names(),
        "num_features": BehaviorFeatureVector.num_features(),
        "use_object_detection": not args.no_yolo,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] Extracted {success}/{len(resolved)} resolved videos to {output_dir}")


if __name__ == "__main__":
    main()
