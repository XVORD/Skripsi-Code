import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ProctorPipeline


def _clip_output_name(video_path: Path, risk_hint: str, suffix: str) -> str:
    parent = video_path.parent.name
    stem = video_path.stem
    safe = f"{risk_hint}_{parent}_{stem}_{suffix}".replace(" ", "_")
    return safe


def run_demo(
    video_path: Path,
    output_dir: Path,
    config: str,
    risk_hint: str,
    max_frames: int | None,
    start_frame: int,
    suffix: str,
    status_source: str,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = _clip_output_name(video_path, risk_hint, suffix)
    video_out = output_dir / f"{base_name}.mp4"
    report_out = output_dir / f"{base_name}.json"
    csv_out = output_dir / f"{base_name}_intervals.csv"

    pipeline = ProctorPipeline(config_path=config)
    pipeline.show_visualization = True
    pipeline._start_time = 0.0
    pipeline._last_timestamp = 0.0
    pipeline._events.clear()
    pipeline._frame_results.clear()
    pipeline.scorer.reset()
    pipeline.feature_extractor.reset()
    pipeline.head_pose.history.clear()
    pipeline.eye_gaze.history.clear()

    if pipeline.enable_object_detection:
        if not pipeline.object_detector.initialize():
            pipeline.enable_object_detection = False

    if not pipeline.video.start(str(video_path)):
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap = pipeline.video.cap
    assert cap is not None
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        pipeline.video.frame_count = start_frame

    fps = pipeline.video.actual_fps or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video writer: {video_out}")

    processed = 0
    try:
        while True:
            if max_frames is not None and processed >= max_frames:
                break
            ret, original, preprocessed = pipeline.video.read_frame_preprocessed()
            if not ret or original is None or preprocessed is None:
                break

            frame_number = pipeline.video.frame_count
            timestamp = frame_number / fps
            pipeline._last_timestamp = timestamp
            data = pipeline._process_frame(original, preprocessed, frame_number, timestamp)
            display = pipeline._draw_visualization(original, data)
            if status_source == "temporal":
                temporal = data.get("temporal")
                label = getattr(temporal, "predicted_label", "normal") if temporal else "normal"
                prob = float(getattr(temporal, "cheat_probability", 0.0)) if temporal else 0.0
                colors = {
                    "normal": (0, 180, 0),
                    "warning": (0, 190, 255),
                    "suspicious": (0, 0, 220),
                }
                color = colors.get(str(label).lower(), (255, 255, 255))
                cv2.rectangle(display, (0, 0), (display.shape[1], 72), (25, 25, 25), -1)
                cv2.putText(
                    display,
                    f"SYSTEM OUTPUT: {str(label).upper()}",
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )
                cv2.putText(
                    display,
                    f"LSTM cheating probability: {prob:.2f} | frame {frame_number} | t={timestamp:.1f}s",
                    (18, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                )
            writer.write(display)
            processed += 1
    finally:
        pipeline.scorer.finalize_interval(timestamp=pipeline._last_timestamp)
        writer.release()
        pipeline.video.stop()
        pipeline.face_detector.release()

    results = pipeline.get_results()
    results["demo"] = {
        "input_video": str(video_path),
        "output_video": str(video_out),
        "start_frame": start_frame,
        "processed_frames": processed,
        "fps": fps,
    }

    intervals = results.get("scoring", {}).get("intervals", [])
    temporal_counts = {}
    for row in intervals:
        label = str(row.get("temporal_prediction") or "normal")
        temporal_counts[label] = temporal_counts.get(label, 0) + 1
    results["demo"]["status_source"] = status_source
    results["demo"]["temporal_prediction_counts"] = temporal_counts

    with report_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    fieldnames = [
        "interval_start",
        "interval_end",
        "total_score",
        "score_percent",
        "risk_level",
        "top_indicators",
        "temporal_prediction",
        "temporal_confidence",
    ]
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        for row in intervals:
            writer_csv.writerow({
                "interval_start": row.get("interval_start"),
                "interval_end": row.get("interval_end"),
                "total_score": row.get("total_score"),
                "score_percent": row.get("score_percent"),
                "risk_level": row.get("risk_level"),
                "top_indicators": "|".join(row.get("top_indicators") or []),
                "temporal_prediction": row.get("temporal_prediction"),
                "temporal_confidence": row.get("temporal_confidence"),
            })

    overall = results.get("scoring", {}).get("overall", {})
    return {
        "input_video": str(video_path),
        "output_video": str(video_out),
        "report": str(report_out),
        "intervals_csv": str(csv_out),
        "start_frame": start_frame,
        "processed_frames": processed,
        "overall_risk": overall.get("risk_level"),
        "avg_score_percent": overall.get("avg_score_percent"),
        "max_score_percent": overall.get("max_score_percent"),
        "risk_distribution": overall.get("risk_distribution", {}),
        "temporal_prediction_counts": temporal_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", default="output/reports/main_system_risk_demo_20260610")
    parser.add_argument("--config", default="configs/config_skripsi_exact.yaml")
    parser.add_argument("--risk-hint", default="demo")
    parser.add_argument("--max-frames", type=int, default=450)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--suffix", default="30s")
    parser.add_argument("--status-source", choices=["risk", "temporal"], default="risk")
    args = parser.parse_args()

    summary = run_demo(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        config=args.config,
        risk_hint=args.risk_hint,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        suffix=args.suffix,
        status_source=args.status_source,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
