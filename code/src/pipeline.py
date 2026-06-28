"""
Pipeline Orchestrator
=====================
Mengintegrasikan semua modul dalam satu pipeline end-to-end:
Video → Face Detection → Head Pose + Eye Gaze → Temporal Modeling → Scoring

Sesuai dengan BAB 3.4 Skripsi: Arsitektur Sistem
"""

import cv2
import numpy as np
import yaml
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime

# Import semua modul
from src.video_acquisition import VideoAcquisition
from src.face_detection import FaceDetector, FaceDetectionResult
from src.object_detection import ObjectDetector
from src.head_pose_estimator import HeadPoseEstimator
from src.eye_gaze_estimator import EyeGazeEstimator
from src.behavior_features import BehaviorFeatureExtractor
from src.temporal_model import TemporalModel
from src.cheating_scorer import CheatingScorer, RiskLevel


class ProctorPipeline:
    """
    Pipeline utama Automated Proctoring System.

    Mengorkestrasi semua modul analisis secara berurutan
    dan menghasilkan skor kecurangan secara real-time.
    """

    RISK_COLORS = {
        RiskLevel.NORMAL: (0, 255, 0),       # Hijau
        RiskLevel.WARNING: (0, 255, 255),    # Kuning
        RiskLevel.SUSPICIOUS: (0, 165, 255),  # Oranye
        RiskLevel.HIGH_RISK: (0, 165, 255),   # Orange
        RiskLevel.CRITICAL: (0, 0, 255),      # Merah
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Inisialisasi semua modul."""
        self.config_path = config_path
        self.config = self._load_config(config_path)

        print("[Pipeline] Initializing modules...")

        # Modul 1: Video Acquisition
        self.video = VideoAcquisition(config_path)

        # Modul 2: Face & Object Detection
        self.face_detector = FaceDetector(config_path)
        self.object_detector = ObjectDetector(config_path)

        # Modul 3: Behavior Analysis
        self.head_pose = HeadPoseEstimator(config_path)
        self.eye_gaze = EyeGazeEstimator(config_path)
        self.feature_extractor = BehaviorFeatureExtractor(config_path)
        # Backward-compatible aliases used by some older tests/scripts.
        self.behavior = self.feature_extractor

        # Modul 4: Temporal Modeling
        self.temporal_model = TemporalModel(config_path)
        # Backward-compatible alias used by some older tests/scripts.
        self.temporal = self.temporal_model

        # Modul 5: Scoring
        self.scorer = CheatingScorer(config_path)

        # State
        self._start_time: float = 0.0
        self._last_timestamp: float = 0.0
        self._events: list = []
        self._frame_results: list = []

        # Settings
        self.show_visualization = True
        self.enable_object_detection = True

        print("[Pipeline] All modules initialized.")

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def run(self, source: Optional[str] = None,
            max_frames: Optional[int] = None,
            show_video: bool = True,
            save_output: bool = False,
            output_path: str = "output/result.avi"):
        """
        Jalankan pipeline proctoring.

        Args:
            source: 'webcam' atau path ke file video
            max_frames: Limit jumlah frame
            show_video: Tampilkan video dengan visualisasi
            save_output: Simpan video output
            output_path: Path output video
        """
        self.show_visualization = show_video
        self._start_time = time.time()
        self._last_timestamp = 0.0
        self._events.clear()
        self._frame_results.clear()
        self.scorer.reset()
        self.feature_extractor.reset()
        self.head_pose.history.clear()
        self.eye_gaze.history.clear()

        # Initialize object detector
        if self.enable_object_detection:
            if not self.object_detector.initialize():
                self.enable_object_detection = False
                print("[Pipeline] Object detection disabled.")

        # Start video capture
        if not self.video.start(source):
            print("[Pipeline] Failed to start video capture.")
            return

        fps = self.video.actual_fps or 15
        video_writer = None

        if save_output:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, fps,
                (self.video.width, self.video.height)
            )

        print(f"\n{'='*60}")
        print(f" AUTOMATED PROCTORING SYSTEM - ACTIVE")
        print(f"{'='*60}")
        print(f" Source: {self.video.source}")
        print(f" Press 'q' to stop")
        print(f"{'='*60}\n")

        try:
            for frame_num, original, preprocessed in self.video.stream_frames(max_frames):
                timestamp = frame_num / fps
                self._last_timestamp = timestamp

                # Process frame through pipeline
                frame_data = self._process_frame(
                    original, preprocessed, frame_num, timestamp
                )

                # Visualize
                if show_video:
                    display = self._draw_visualization(original, frame_data)
                    cv2.imshow("Proctor System", display)

                    if save_output and video_writer:
                        video_writer.write(display)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('m'):  # Toggle mesh
                        self._toggle_mesh = not getattr(self, '_toggle_mesh', False)

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted by user.")

        finally:
            # Close any unfinished interval so short sessions still produce scores.
            self.scorer.finalize_interval(timestamp=self._last_timestamp)

            # Cleanup
            self.video.stop()
            self.face_detector.release()
            if video_writer:
                video_writer.release()
            if show_video:
                cv2.destroyAllWindows()

            # Print summary
            self._print_summary()

    def _process_frame(self, original_bgr: np.ndarray,
                       preprocessed_rgb: np.ndarray,
                       frame_number: int,
                       timestamp: float) -> Dict:
        """
        Proses satu frame melalui seluruh pipeline.

        Returns:
            Dictionary berisi semua hasil analisis
        """
        h, w = original_bgr.shape[:2]
        data = {
            "frame_number": frame_number,
            "timestamp": timestamp,
        }

        # ── Modul 2a: Face Detection ──
        face_result = self.face_detector.detect(preprocessed_rgb, frame_number, timestamp)
        data["face"] = face_result

        # ── Modul 2b: Object Detection ──
        obj_result = None
        if self.enable_object_detection:
            obj_result = self.object_detector.detect(original_bgr, frame_number, timestamp)
        data["objects"] = obj_result

        # ── Modul 3: Behavior Analysis ──
        head_pose_result = None
        gaze_result = None

        if face_result.face_present and len(face_result.faces) > 0:
            primary_face = face_result.faces[0]

            # Head Pose
            head_pose_result = self.head_pose.estimate(
                primary_face["pose_landmarks_2d"], w, h, frame_number, timestamp
            )

            # Eye Gaze
            gaze_result = self.eye_gaze.estimate(
                primary_face, frame_number, timestamp, head_pose_result
            )

        data["head_pose"] = head_pose_result
        data["gaze"] = gaze_result

        # ── Feature Extraction ──
        features = self.feature_extractor.extract(
            face_result, head_pose_result, gaze_result,
            obj_result, frame_number, timestamp
        )
        data["features"] = features

        # ── Modul 4: Temporal Modeling ──
        temporal_pred = None
        feature_seq = self.feature_extractor.get_feature_sequence(
            self.temporal_model.window_size
        )
        if feature_seq is not None:
            temporal_pred = self.temporal_model.predict(
                feature_seq, frame_number, timestamp
            )
        data["temporal"] = temporal_pred

        # ── Modul 5: Scoring ──
        head_stats = self.head_pose.get_movement_stats()
        fps = max(float(self.video.actual_fps or self.config.get("video", {}).get("fps", 15) or 15), 1.0)
        window_30f_seconds = float(self.temporal_model.window_size) / fps
        head_stats_30f = self.head_pose.get_movement_stats(window_seconds=window_30f_seconds)
        interval_score = self.scorer.add_frame_data(
            timestamp=timestamp,
            face_absent=face_result.face_absent,
            multiple_faces=face_result.multiple_faces,
            offscreen_gaze=gaze_result.is_offscreen if gaze_result else False,
            suspicious_object=obj_result.has_suspicious_object if obj_result else False,
            reading_pattern=gaze_result.is_reading_pattern if gaze_result else False,
            looking_away=head_pose_result.is_looking_away if head_pose_result else False,
            head_movement_stats=head_stats,
            head_movement_stats_30f=head_stats_30f,
            temporal_prediction=temporal_pred,
        )
        data["interval_score"] = interval_score

        # Log events
        self._log_events(data)
        self._frame_results.append(data)

        return data

    def _log_events(self, data: Dict):
        """Log peristiwa penting."""
        ts = data["timestamp"]
        face = data["face"]
        gaze = data["gaze"]
        obj = data.get("objects")
        score = data.get("interval_score")

        def add_event(event_type: str, message: str):
            self._events.append({
                "timestamp": ts,
                "time": ts,  # backward compatibility
                "type": event_type,
                "message": message,
                "msg": message,  # backward compatibility
            })

        if face.face_absent:
            add_event("face_absent", "Face absent")
        if face.multiple_faces:
            add_event("multiple_faces", f"{face.num_faces} faces detected")
        if gaze and gaze.is_reading_pattern:
            add_event("reading", "Reading pattern detected")
        if obj and obj.has_suspicious_object:
            names = [o.class_name for o in obj.suspicious_objects]
            add_event("object", f"Objects: {names}")
        if score and score.risk_level in (RiskLevel.WARNING, RiskLevel.SUSPICIOUS, RiskLevel.HIGH_RISK, RiskLevel.CRITICAL):
            add_event("risk_alert", f"Score: {score.total_score:.2f} [{score.risk_level.value}]")

    def _draw_visualization(self, frame_bgr: np.ndarray, data: Dict) -> np.ndarray:
        """Gambar semua visualisasi pada frame."""
        display = frame_bgr.copy()
        h, w = display.shape[:2]

        face = data["face"]
        head_pose = data.get("head_pose")
        gaze = data.get("gaze")
        obj = data.get("objects")
        temporal = data.get("temporal")
        interval_score = data.get("interval_score")

        # Face detection overlay
        display = self.face_detector.draw_detection(display, face, draw_mesh=False)

        # Head pose axes
        if head_pose and head_pose.is_valid:
            display = self.head_pose.draw_pose(display, head_pose)

        # Gaze visualization
        if gaze and gaze.is_valid:
            display = self.eye_gaze.draw_gaze(display, gaze)

        # Object detection
        if obj:
            display = self.object_detector.draw_detection(display, obj)

        # ── Status Panel (top-right) ──
        panel_x = w - 300
        panel_y = 10

        # Current risk level
        latest_scores = list(self.scorer.score_history)
        if latest_scores:
            last_score = latest_scores[-1]
            risk = last_score.risk_level
            color = self.RISK_COLORS[risk]
            cv2.rectangle(display, (panel_x - 5, panel_y - 5),
                          (w - 5, panel_y + 100), (40, 40, 40), -1)
            cv2.putText(display, f"Risk: {risk.value.upper()}",
                        (panel_x, panel_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, f"Score: {last_score.total_score:.2f}",
                        (panel_x, panel_y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Score%: {last_score.score_percent:.1f}%",
                        (panel_x, panel_y + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
            if last_score.top_indicators:
                cv2.putText(display, f"Top: {', '.join(last_score.top_indicators[:2])}",
                            (panel_x, panel_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Temporal prediction
        if temporal and temporal.is_cheating_predicted:
            cv2.putText(display, f"AI: {temporal.predicted_label} ({temporal.confidence:.0%})",
                        (panel_x, panel_y + 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # Frame counter
        cv2.putText(display, f"Frame: {data['frame_number']}  t={data['timestamp']:.1f}s",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        return display

    def _print_summary(self):
        """Print summary setelah selesai."""
        overall = self.scorer.get_overall_score()

        print(f"\n{'='*60}")
        print(f" SESSION SUMMARY")
        print(f"{'='*60}")
        print(f" Total Frames: {self.video.frame_count}")
        print(f" Duration: {self.video.frame_count / max(self.video.actual_fps, 1):.1f}s")
        print(f" Total Intervals: {overall.get('total_intervals', 0)}")
        print(f" Average Score: {overall.get('avg_score', 0):.3f}")
        print(f" Average Score (%): {overall.get('avg_score_percent', 0):.1f}%")
        print(f" Max Score: {overall.get('max_score', 0):.3f}")
        print(f" Max Score (%): {overall.get('max_score_percent', 0):.1f}%")
        print(f" Overall Risk: {overall.get('overall_risk', 'N/A')}")

        if overall.get("risk_distribution"):
            print(f"\n Risk Distribution:")
            for level, count in overall["risk_distribution"].items():
                print(f"   {level}: {count}")

        events_by_type = {}
        for e in self._events:
            t = e["type"]
            events_by_type[t] = events_by_type.get(t, 0) + 1
        if events_by_type:
            print(f"\n Events Detected:")
            for etype, count in events_by_type.items():
                print(f"   {etype}: {count}")

        print(f"{'='*60}\n")

    def get_results(self) -> Dict:
        """Dapatkan semua hasil dalam format JSON-serializable."""
        overall = self.scorer.get_overall_score()
        fps = self.video.actual_fps or 0.0
        duration_seconds = self.video.frame_count / max(fps, 1.0)

        intervals = [
            {
                "start": s.interval_start,
                "end": s.interval_end,
                "score": s.total_score,
                "score_percent": s.score_percent,
                "risk": s.risk_level.value,
                "indicators": s.indicator_scores,
                "top": s.top_indicators,
                "temporal_pred": s.temporal_prediction,
                "temporal_confidence": s.temporal_confidence,
                "temporal_cheat_probability": s.temporal_confidence,
                "prob_third_party": s.prob_third_party,
                "prob_ai_assistance": s.prob_ai_assistance,
            }
            for s in self.scorer.score_history
        ]

        scoring_intervals = [
            {
                "interval_start": iv["start"],
                "interval_end": iv["end"],
                "total_score": iv["score"],
                "score_percent": iv["score_percent"],
                "risk_level": iv["risk"],
                "indicator_scores": iv["indicators"],
                "top_indicators": iv["top"],
                "temporal_prediction": iv["temporal_pred"],
                "temporal_confidence": iv["temporal_confidence"],
                "temporal_cheat_probability": iv["temporal_cheat_probability"],
                "prob_third_party": iv["prob_third_party"],
                "prob_ai_assistance": iv["prob_ai_assistance"],
            }
            for iv in intervals
        ]

        cheating_timeline = self._build_cheating_timeline(scoring_intervals)
        probability_summary = self._build_temporal_probability_summary(overall)

        session_block = {
            "source": self.video.source,
            "total_frames": self.video.frame_count,
            "fps": fps,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat(),
        }

        return {
            # Current schema used by Streamlit dashboard.
            "session": session_block,
            "scoring": {
                "overall": {
                    "overall_score": overall.get("overall_score", 0.0),
                    "overall_score_percent": overall.get("overall_score_percent", 0.0),
                    "risk_level": overall.get("overall_risk", RiskLevel.NORMAL.value),
                    "max_score": overall.get("max_score", 0.0),
                    "max_score_percent": overall.get("max_score_percent", 0.0),
                    "avg_score": overall.get("avg_score", 0.0),
                    "avg_score_percent": overall.get("avg_score_percent", 0.0),
                    "total_intervals": overall.get("total_intervals", 0),
                    "risk_distribution": overall.get("risk_distribution", {}),
                    "cheating_probabilities": probability_summary,
                },
                "intervals": scoring_intervals,
                "cheating_timeline": cheating_timeline,
                "thresholds": self.scorer.get_thresholds(),
            },
            # Legacy schema retained for backward compatibility.
            "session_info": session_block,
            "overall_score": overall,
            "events": self._events,
            "intervals": intervals,
            "cheating_timeline": cheating_timeline,
        }

    @staticmethod
    def _fmt_mmss(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    @staticmethod
    def _fmt_hhmmss(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _classify_interval_cheating_type(self, iv: Dict) -> Optional[Dict]:
        """
        Heuristic cheating-type classifier per interval.
        Returns None for intervals below suspicious threshold.
        """
        score = float(iv.get("total_score", 0.0))
        if score < float(self.scorer.threshold_suspicious):
            return None

        inds = iv.get("indicator_scores", {}) or {}
        multiple_faces = float(inds.get("multiple_faces", 0.0))
        face_absent = float(inds.get("face_absent", 0.0))
        head_move = float(inds.get("head_movement_anomaly", 0.0))
        suspicious_object = float(inds.get("suspicious_object", 0.0))
        reading_pattern = float(inds.get("reading_pattern", 0.0))
        offscreen = float(inds.get("offscreen_gaze", 0.0))
        temporal = float(inds.get("temporal_anomaly", 0.0))

        # Third-party signature: extra face/absence + side interactions.
        third_party_score = (
            0.35 * multiple_faces
            + 0.25 * face_absent
            + 0.20 * head_move
            + 0.20 * suspicious_object
        )

        # AI-assistance signature: reading/offscreen/temporal pattern.
        ai_assistance_score = (
            0.45 * reading_pattern
            + 0.30 * offscreen
            + 0.15 * temporal
            + 0.10 * suspicious_object
        )

        if ai_assistance_score >= third_party_score and ai_assistance_score >= 0.20:
            ctype = "ai_assistance"
            cscore = ai_assistance_score
        elif third_party_score > ai_assistance_score and third_party_score >= 0.20:
            ctype = "third_party_assistance"
            cscore = third_party_score
        else:
            ctype = "suspicious_general"
            cscore = score

        reasons = sorted(inds.items(), key=lambda kv: kv[1], reverse=True)
        reasons = [k for k, v in reasons if float(v) > 0.1][:3]
        return {
            "type": ctype,
            "type_score": float(np.clip(cscore, 0.0, 1.0)),
            "reasons": reasons,
        }

    def _build_cheating_timeline(self, scoring_intervals: List[Dict]) -> List[Dict]:
        """Build merged timestamp ranges per cheating type for reporting."""
        timeline: List[Dict] = []
        active = None

        for iv in scoring_intervals:
            cls = self._classify_interval_cheating_type(iv)
            if cls is None:
                if active is not None:
                    timeline.append(active)
                    active = None
                continue

            start = float(iv.get("interval_start", 0.0))
            end = float(iv.get("interval_end", start))
            ctype = cls["type"]

            if active is None:
                active = {
                    "type": ctype,
                    "start_seconds": start,
                    "end_seconds": end,
                    "interval_count": 1,
                    "avg_type_score": cls["type_score"],
                    "reasons": list(cls["reasons"]),
                }
                continue

            same_type = active["type"] == ctype
            contiguous = start <= float(active["end_seconds"]) + 1e-6
            if same_type and contiguous:
                n = int(active["interval_count"])
                active["end_seconds"] = end
                active["interval_count"] = n + 1
                active["avg_type_score"] = float(
                    (active["avg_type_score"] * n + cls["type_score"]) / (n + 1)
                )
                for r in cls["reasons"]:
                    if r not in active["reasons"] and len(active["reasons"]) < 3:
                        active["reasons"].append(r)
            else:
                timeline.append(active)
                active = {
                    "type": ctype,
                    "start_seconds": start,
                    "end_seconds": end,
                    "interval_count": 1,
                    "avg_type_score": cls["type_score"],
                    "reasons": list(cls["reasons"]),
                }

        if active is not None:
            timeline.append(active)

        for item in timeline:
            start = float(item["start_seconds"])
            end = float(item["end_seconds"])
            item["start_mmss"] = self._fmt_mmss(start)
            item["end_mmss"] = self._fmt_mmss(end)
            item["start_hhmmss"] = self._fmt_hhmmss(start)
            item["end_hhmmss"] = self._fmt_hhmmss(end)
            item["range_mmss"] = f"{item['start_mmss']}-{item['end_mmss']}"
            item["range_hhmmss"] = f"{item['start_hhmmss']}-{item['end_hhmmss']}"

        return timeline

    def _build_temporal_probability_summary(self, overall: Dict) -> Dict[str, float]:
        """Aggregate coarse cheating-type probabilities for reporting UI."""
        if not self.scorer.score_history:
            return {
                "normal": 1.0,
                "third_party": 0.0,
                "ai_assistance": 0.0,
                "suspicious": 0.0,
            }

        # In binary temporal mode, temporal_confidence stores cheating probability.
        suspicious = float(np.mean([s.temporal_confidence for s in self.scorer.score_history]))
        third_party = float(np.mean([s.prob_third_party for s in self.scorer.score_history]))
        ai_assistance = float(np.mean([s.prob_ai_assistance for s in self.scorer.score_history]))
        normal = max(0.0, 1.0 - suspicious)

        probs = {
            "normal": normal,
            "third_party": third_party,
            "ai_assistance": ai_assistance,
            "suspicious": suspicious,
        }
        total = sum(probs.values())
        if total <= 0:
            return {
                "normal": 1.0,
                "third_party": 0.0,
                "ai_assistance": 0.0,
                "suspicious": 0.0,
            }
        return {k: float(v / total) for k, v in probs.items()}

    def save_results(self, path: str = "output/reports/session_report.json"):
        """Simpan hasil ke file JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        results = self.get_results()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[Pipeline] Results saved to {path}")


# === Entry Point ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Automated Proctoring System")
    parser.add_argument("--source", default="webcam", help="Video source: 'webcam' or path to video file")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--no-video", action="store_true", help="Disable video display")
    parser.add_argument("--save-video", action="store_true", help="Save output video")
    parser.add_argument("--save-report", action="store_true", help="Save JSON report")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")

    args = parser.parse_args()

    pipeline = ProctorPipeline(config_path=args.config)
    pipeline.run(
        source=args.source,
        max_frames=args.max_frames,
        show_video=not args.no_video,
        save_output=args.save_video,
    )

    if args.save_report:
        pipeline.save_results()
