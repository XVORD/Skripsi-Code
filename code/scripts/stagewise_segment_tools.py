"""
Stage-Wise Segment Annotation Tools
===================================
Utilities untuk anotasi stage-wise berbasis rentang segmen (bukan per-frame).

Workflow:
1) Generate template per-segment dari data/ground_truth:
   python scripts/stagewise_segment_tools.py template \
     --gt-root data/ground_truth \
     --output data/annotations/stage_eval_segment_template.csv

2) Isi kolom *_gt per segmen (manual annotator):
   face_present_gt, multiple_faces_gt, offscreen_gaze_gt, looking_away_gt,
   suspicious_object_gt, yaw_gt, pitch_gt, roll_gt, gaze_x_gt, gaze_y_gt

3) Cek progres anotasi:
   python scripts/stagewise_segment_tools.py status \
     --annotations data/annotations/stage_eval_segment_template.csv

4) Konversi ke frame-level CSV (sampling dalam range):
   python scripts/stagewise_segment_tools.py expand \
     --segments data/annotations/stage_eval_segment_template.csv \
     --output data/annotations/stage_eval_from_segments.csv \
     --sample-every 30 \
     --buffer-seconds 0.5

5) Evaluasi pakai evaluator existing:
   python scripts/eval_stagewise.py evaluate \
     --annotations data/annotations/stage_eval_from_segments.csv \
     --videos-root data/videos
"""

import csv
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional


BINARY_GT_COLUMNS = [
    "face_present_gt",
    "multiple_faces_gt",
    "offscreen_gaze_gt",
    "looking_away_gt",
    "suspicious_object_gt",
]

REGRESSION_GT_COLUMNS = [
    "yaw_gt",
    "pitch_gt",
    "roll_gt",
    "gaze_x_gt",
    "gaze_y_gt",
]

SEGMENT_COLUMNS = [
    "video",
    "scenario_gt",
    "global_label_gt",
    "global_label_name_gt",
    "segment_index",
    "segment_name_gt",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "event_start_time",
    "event_end_time",
    "duration_seconds",
    "fps",
    "face_present_gt",
    "multiple_faces_gt",
    "offscreen_gaze_gt",
    "looking_away_gt",
    "suspicious_object_gt",
    "yaw_gt",
    "pitch_gt",
    "roll_gt",
    "gaze_x_gt",
    "gaze_y_gt",
    "notes",
]

FRAME_COLUMNS = [
    "video",
    "frame",
    "timestamp",
    "face_present_gt",
    "multiple_faces_gt",
    "offscreen_gaze_gt",
    "looking_away_gt",
    "suspicious_object_gt",
    "yaw_gt",
    "pitch_gt",
    "roll_gt",
    "gaze_x_gt",
    "gaze_y_gt",
    "notes",
]


def _to_int_binary(value: str) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s == "":
        return None
    if s in ("1", "true", "yes", "y"):
        return 1
    if s in ("0", "false", "no", "n"):
        return 0
    # Tolerate minor manual-annotation typos like "0-" / "1," / "0 ;".
    m = re.match(r"^\s*([01])(?:\D.*)?$", s)
    if m:
        return int(m.group(1))
    return None


def _to_float(value: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_binary_cell(value: str) -> str:
    parsed = _to_int_binary(value)
    return "" if parsed is None else str(int(parsed))


def generate_segment_template(gt_root: str, output_csv: str):
    gt_root_path = Path(gt_root)
    out_path = Path(output_csv)
    _ensure_parent(out_path)

    label_name_map = {0: "normal", 1: "third_party", 2: "ai_assistance"}
    rows: List[Dict[str, str]] = []

    for json_path in sorted(gt_root_path.rglob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            gt = json.load(f)

        scenario = str(gt.get("scenario", json_path.parent.name))
        filename = str(gt.get("filename", json_path.stem))
        video_rel = f"{scenario}/{filename}.mp4"
        fps = float(gt.get("fps", 15))
        segments = gt.get("segments", [])

        for seg in segments:
            label = int(seg.get("label", 0))
            start_frame = int(seg.get("start_frame", 0))
            end_frame = int(seg.get("end_frame", start_frame))
            start_time = float(seg.get("start_time", start_frame / fps))
            end_time = float(seg.get("end_time", (end_frame + 1) / fps))
            duration_seconds = float(seg.get("duration_seconds", max(0.0, end_time - start_time)))

            row = {
                "video": video_rel,
                "scenario_gt": scenario,
                "global_label_gt": str(label),
                "global_label_name_gt": label_name_map.get(label, f"label_{label}"),
                "segment_index": str(int(seg.get("index", 0))),
                "segment_name_gt": str(seg.get("name", "")),
                "start_frame": str(start_frame),
                "end_frame": str(end_frame),
                "start_time": f"{start_time:.3f}",
                "end_time": f"{end_time:.3f}",
                # Default event range = full segment.
                # Annotator can narrow this if suspicious behavior only appears
                # in part of the segment.
                "event_start_time": f"{start_time:.3f}",
                "event_end_time": f"{end_time:.3f}",
                "duration_seconds": f"{duration_seconds:.3f}",
                "fps": f"{fps:.3f}",
                "face_present_gt": "",
                "multiple_faces_gt": "",
                "offscreen_gaze_gt": "",
                "looking_away_gt": "",
                "suspicious_object_gt": "",
                "yaw_gt": "",
                "pitch_gt": "",
                "roll_gt": "",
                "gaze_x_gt": "",
                "gaze_y_gt": "",
                "notes": "",
            }
            rows.append(row)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SEGMENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[OK] Segment template written: {out_path}")
    print(f"     Rows (segments): {len(rows)}")


def status(annotations_csv: str):
    ann_path = Path(annotations_csv)
    with open(ann_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    total_rows = len(rows)
    binary_ready = 0
    reg_ready = 0
    per_col = {}

    for col in BINARY_GT_COLUMNS:
        filled = sum(1 for r in rows if _to_int_binary(r.get(col)) is not None)
        per_col[col] = (filled, total_rows)
    for col in REGRESSION_GT_COLUMNS:
        filled = sum(1 for r in rows if _to_float(r.get(col)) is not None)
        per_col[col] = (filled, total_rows)

    for r in rows:
        if all(_to_int_binary(r.get(c)) is not None for c in BINARY_GT_COLUMNS):
            binary_ready += 1
        if all(_to_float(r.get(c)) is not None for c in REGRESSION_GT_COLUMNS):
            reg_ready += 1

    print(f"[STATUS] {ann_path}")
    print(f"  Total segment rows: {total_rows}")
    print(f"  Binary-ready rows:     {binary_ready}/{total_rows} ({(binary_ready/total_rows if total_rows else 0):.2%})")
    print(f"  Regression-ready rows: {reg_ready}/{total_rows} ({(reg_ready/total_rows if total_rows else 0):.2%})")
    print("")
    print("  Per-column fill rate:")
    for col in BINARY_GT_COLUMNS + REGRESSION_GT_COLUMNS:
        filled, total = per_col[col]
        rate = filled / total if total else 0.0
        print(f"    - {col:20s}: {filled:4d}/{total:<4d} ({rate:.2%})")


def expand_segments_to_frames(segments_csv: str, output_csv: str, sample_every: int = 30):
    seg_path = Path(segments_csv)
    out_path = Path(output_csv)
    _ensure_parent(out_path)

    with open(seg_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    frame_rows: List[Dict[str, str]] = []
    for r in rows:
        start_frame = int(float(r.get("start_frame", 0)))
        end_frame = int(float(r.get("end_frame", start_frame)))
        fps = float(r.get("fps", 15.0))
        if fps <= 0:
            fps = 15.0
        step = max(1, int(sample_every))

        indices = list(range(start_frame, end_frame + 1, step))
        if not indices or indices[-1] != end_frame:
            indices.append(end_frame)

        seg_name = r.get("segment_name_gt", "")
        global_name = r.get("global_label_name_gt", "")
        base_notes = (r.get("notes") or "").strip()
        suffix = f"segment={seg_name};global={global_name}"
        notes = f"{base_notes} | {suffix}".strip(" |")

        seg_start_t = float(r.get("start_time", start_frame / fps))
        seg_end_t = float(r.get("end_time", (end_frame + 1) / fps))
        event_start_t = _to_float(r.get("event_start_time"))
        event_end_t = _to_float(r.get("event_end_time"))
        if event_start_t is None:
            event_start_t = seg_start_t
        if event_end_t is None:
            event_end_t = seg_end_t
        if event_end_t < event_start_t:
            event_start_t, event_end_t = event_end_t, event_start_t

        for frame_idx in indices:
            ts = frame_idx / fps
            # event labels only valid inside [event_start_t, event_end_t]
            in_event = (event_start_t <= ts <= event_end_t)
            frame_rows.append({
                "video": r.get("video", ""),
                "frame": str(frame_idx),
                "timestamp": f"{ts:.3f}",
                "face_present_gt": _normalize_binary_cell(r.get("face_present_gt", "")) if in_event else "",
                "multiple_faces_gt": _normalize_binary_cell(r.get("multiple_faces_gt", "")) if in_event else "",
                "offscreen_gaze_gt": _normalize_binary_cell(r.get("offscreen_gaze_gt", "")) if in_event else "",
                "looking_away_gt": _normalize_binary_cell(r.get("looking_away_gt", "")) if in_event else "",
                "suspicious_object_gt": _normalize_binary_cell(r.get("suspicious_object_gt", "")) if in_event else "",
                "yaw_gt": r.get("yaw_gt", "") if in_event else "",
                "pitch_gt": r.get("pitch_gt", "") if in_event else "",
                "roll_gt": r.get("roll_gt", "") if in_event else "",
                "gaze_x_gt": r.get("gaze_x_gt", "") if in_event else "",
                "gaze_y_gt": r.get("gaze_y_gt", "") if in_event else "",
                "notes": notes if in_event else f"{notes} | skipped_outside_event",
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow(row)

    print(f"[OK] Expanded frame-level CSV written: {out_path}")
    print(f"     Rows (frames sampled): {len(frame_rows)}")


def expand_segments_to_frames_with_buffer(
    segments_csv: str,
    output_csv: str,
    sample_every: int = 30,
    buffer_seconds: float = 0.5,
):
    """
    Expand segment annotations with transition buffer.

    Labels are only applied for timestamps in:
      [event_start_time + buffer_seconds, event_end_time - buffer_seconds]
    Outside this inner range, GT cells are left blank.
    """
    seg_path = Path(segments_csv)
    out_path = Path(output_csv)
    _ensure_parent(out_path)

    with open(seg_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]

    frame_rows: List[Dict[str, str]] = []
    b = max(0.0, float(buffer_seconds))
    for r in rows:
        start_frame = int(float(r.get("start_frame", 0)))
        end_frame = int(float(r.get("end_frame", start_frame)))
        fps = float(r.get("fps", 15.0))
        if fps <= 0:
            fps = 15.0
        step = max(1, int(sample_every))

        indices = list(range(start_frame, end_frame + 1, step))
        if not indices or indices[-1] != end_frame:
            indices.append(end_frame)

        seg_name = r.get("segment_name_gt", "")
        global_name = r.get("global_label_name_gt", "")
        base_notes = (r.get("notes") or "").strip()
        suffix = f"segment={seg_name};global={global_name}"
        notes = f"{base_notes} | {suffix}".strip(" |")

        seg_start_t = float(r.get("start_time", start_frame / fps))
        seg_end_t = float(r.get("end_time", (end_frame + 1) / fps))
        event_start_t = _to_float(r.get("event_start_time"))
        event_end_t = _to_float(r.get("event_end_time"))
        if event_start_t is None:
            event_start_t = seg_start_t
        if event_end_t is None:
            event_end_t = seg_end_t
        if event_end_t < event_start_t:
            event_start_t, event_end_t = event_end_t, event_start_t

        inner_start = event_start_t + b
        inner_end = event_end_t - b
        if inner_end < inner_start:
            # Very short event: fall back to full event range.
            inner_start = event_start_t
            inner_end = event_end_t

        for frame_idx in indices:
            ts = frame_idx / fps
            in_event = (inner_start <= ts <= inner_end)
            frame_rows.append({
                "video": r.get("video", ""),
                "frame": str(frame_idx),
                "timestamp": f"{ts:.3f}",
                "face_present_gt": _normalize_binary_cell(r.get("face_present_gt", "")) if in_event else "",
                "multiple_faces_gt": _normalize_binary_cell(r.get("multiple_faces_gt", "")) if in_event else "",
                "offscreen_gaze_gt": _normalize_binary_cell(r.get("offscreen_gaze_gt", "")) if in_event else "",
                "looking_away_gt": _normalize_binary_cell(r.get("looking_away_gt", "")) if in_event else "",
                "suspicious_object_gt": _normalize_binary_cell(r.get("suspicious_object_gt", "")) if in_event else "",
                "yaw_gt": r.get("yaw_gt", "") if in_event else "",
                "pitch_gt": r.get("pitch_gt", "") if in_event else "",
                "roll_gt": r.get("roll_gt", "") if in_event else "",
                "gaze_x_gt": r.get("gaze_x_gt", "") if in_event else "",
                "gaze_y_gt": r.get("gaze_y_gt", "") if in_event else "",
                "notes": notes if in_event else f"{notes} | skipped_buffer_or_outside_event",
            })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FRAME_COLUMNS)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow(row)

    print(f"[OK] Expanded frame-level CSV with buffer written: {out_path}")
    print(f"     Rows (frames sampled): {len(frame_rows)}")
    print(f"     Buffer seconds: {b:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Stage-wise segment annotation tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tpl = sub.add_parser("template", help="Generate per-segment annotation template")
    p_tpl.add_argument("--gt-root", default="data/ground_truth")
    p_tpl.add_argument("--output", default="data/annotations/stage_eval_segment_template.csv")

    p_status = sub.add_parser("status", help="Check annotation completion status")
    p_status.add_argument("--annotations", required=True)

    p_expand = sub.add_parser("expand", help="Expand per-segment annotations to frame-level CSV")
    p_expand.add_argument("--segments", required=True, help="Segment annotation CSV")
    p_expand.add_argument("--output", default="data/annotations/stage_eval_from_segments.csv")
    p_expand.add_argument("--sample-every", type=int, default=30,
                          help="Sample every N frames within each segment")
    p_expand.add_argument("--buffer-seconds", type=float, default=0.0,
                          help="Skip transition edges by this many seconds on both sides")

    args = parser.parse_args()

    if args.cmd == "template":
        generate_segment_template(args.gt_root, args.output)
        return
    if args.cmd == "status":
        status(args.annotations)
        return
    if args.cmd == "expand":
        if float(args.buffer_seconds) > 0:
            expand_segments_to_frames_with_buffer(
                args.segments,
                args.output,
                sample_every=args.sample_every,
                buffer_seconds=float(args.buffer_seconds),
            )
        else:
            expand_segments_to_frames(args.segments, args.output, sample_every=args.sample_every)
        return


if __name__ == "__main__":
    main()
