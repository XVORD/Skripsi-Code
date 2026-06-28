"""
Modul Ekstraksi Fitur Perilaku
===============================
Menggabungkan fitur dari head pose dan eye gaze menjadi satu feature vector
per frame, serta menghitung fitur statistik rolling window.

Sesuai dengan BAB 2.6.2 & 3.4.3 Skripsi:
- Integrasi Head Pose dan Eye Gaze untuk Deteksi Kecurangan
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque


@dataclass
class BehaviorFeatureVector:
    """Feature vector gabungan untuk satu frame."""
    frame_number: int = 0
    timestamp: float = 0.0

    # Head Pose features
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Eye Gaze features
    gaze_x: float = 0.0
    gaze_y: float = 0.0

    # Flags & structured event features (BAB 3.4.4)
    looking_away: float = 0.0       # 0 or 1
    offscreen_gaze: float = 0.0     # 0 or 1
    is_fixating: float = 0.0        # 0 or 1
    is_saccade: float = 0.0         # 0 or 1
    num_faces: float = 0.0          # count
    suspicious_object: float = 0.0  # 0 or 1

    # Eye openness ratios (EAR)
    left_ear: float = 0.0
    right_ear: float = 0.0

    # Motion dynamics
    yaw_velocity: float = 0.0
    gaze_velocity: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert ke numpy array untuk input ke model."""
        return np.array([
            self.yaw, self.pitch, self.roll,
            self.gaze_x, self.gaze_y,
            self.looking_away, self.offscreen_gaze,
            self.is_fixating, self.is_saccade,
            self.num_faces,
            self.suspicious_object,
            self.left_ear, self.right_ear,
            self.yaw_velocity, self.gaze_velocity,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Nama-nama fitur dalam urutan yang sama dengan to_array()."""
        return [
            "yaw", "pitch", "roll",
            "gaze_x", "gaze_y",
            "looking_away", "offscreen_gaze",
            "is_fixating", "is_saccade",
            "num_faces",
            "suspicious_object",
            "left_ear", "right_ear",
            "yaw_velocity", "gaze_velocity",
        ]

    @staticmethod
    def num_features() -> int:
        return 15


class BehaviorFeatureExtractor:
    """
    Ekstraktor fitur perilaku gabungan.

    Menggabungkan output dari:
    - FaceDetector (face presence, multi-face, landmarks)
    - HeadPoseEstimator (yaw, pitch, roll)
    - EyeGazeEstimator (gaze direction, fixation, saccade, reading)
    - ObjectDetector (suspicious objects)

    Menghasilkan feature vector per frame dan rolling window statistics.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        config = self._load_config(config_path)
        self.fps = config.get("video", {}).get("fps", 15)

        # History
        self.history: deque = deque(maxlen=600)  # ~40 seconds

        # Rolling window sizes (in seconds)
        self.window_sizes = [5.0, 10.0, 30.0]
        self._prev_yaw: Optional[float] = None
        self._prev_gaze: Optional[np.ndarray] = None
        self._prev_ts: Optional[float] = None

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def extract(self, face_result, head_pose_result, gaze_result,
                object_result=None, frame_number: int = 0,
                timestamp: float = 0.0) -> BehaviorFeatureVector:
        """
        Gabungkan semua hasil analisis menjadi satu feature vector.

        Args:
            face_result: FaceDetectionResult
            head_pose_result: HeadPoseResult
            gaze_result: GazeResult
            object_result: ObjectDetectionResult (opsional)
            frame_number: Nomor frame
            timestamp: Timestamp

        Returns:
            BehaviorFeatureVector
        """
        fv = BehaviorFeatureVector(
            frame_number=frame_number,
            timestamp=timestamp,
        )

        # Face/object features
        if face_result:
            fv.num_faces = float(face_result.num_faces)
        if object_result:
            fv.suspicious_object = 1.0 if object_result.has_suspicious_object else 0.0

        # Head pose features
        if head_pose_result and head_pose_result.is_valid:
            fv.yaw = head_pose_result.yaw
            fv.pitch = head_pose_result.pitch
            fv.roll = head_pose_result.roll
            fv.looking_away = 1.0 if head_pose_result.is_looking_away else 0.0

            if self._prev_yaw is not None and self._prev_ts is not None:
                dt = max(timestamp - self._prev_ts, 1.0 / max(self.fps, 1))
                fv.yaw_velocity = abs(fv.yaw - self._prev_yaw) / dt

        # Eye gaze features
        if gaze_result and gaze_result.is_valid:
            fv.gaze_x = gaze_result.gaze_x
            fv.gaze_y = gaze_result.gaze_y
            fv.offscreen_gaze = 1.0 if gaze_result.is_offscreen else 0.0
            fv.is_fixating = 1.0 if gaze_result.is_fixating else 0.0
            fv.is_saccade = 1.0 if gaze_result.is_saccade else 0.0
            fv.left_ear = gaze_result.left_eye_ratio
            fv.right_ear = gaze_result.right_eye_ratio

            if self._prev_gaze is not None and self._prev_ts is not None:
                dt = max(timestamp - self._prev_ts, 1.0 / max(self.fps, 1))
                gaze_delta = np.array([fv.gaze_x, fv.gaze_y], dtype=np.float32) - self._prev_gaze
                fv.gaze_velocity = float(np.linalg.norm(gaze_delta) / dt)

        # Update previous valid values for next-frame velocity calculation.
        if head_pose_result and head_pose_result.is_valid:
            self._prev_yaw = fv.yaw
            self._prev_ts = timestamp
        if gaze_result and gaze_result.is_valid:
            self._prev_gaze = np.array([fv.gaze_x, fv.gaze_y], dtype=np.float32)
            self._prev_ts = timestamp

        # Store
        self.history.append(fv)

        return fv

    def get_rolling_stats(self, window_seconds: float = 5.0) -> Dict:
        """
        Hitung statistik rolling window dari feature history.

        Args:
            window_seconds: Ukuran window (detik)

        Returns:
            Dictionary berisi statistik per fitur
        """
        if len(self.history) < 2:
            return {"valid": False}

        current_time = self.history[-1].timestamp
        start_time = current_time - window_seconds

        windowed = [fv for fv in self.history if fv.timestamp >= start_time]
        if len(windowed) < 2:
            return {"valid": False}

        # Convert ke matrix
        matrix = np.array([fv.to_array() for fv in windowed])
        names = BehaviorFeatureVector.feature_names()

        stats = {
            "valid": True,
            "window_seconds": window_seconds,
            "num_samples": len(windowed),
        }

        # Stats per fitur
        for i, name in enumerate(names):
            col = matrix[:, i]
            stats[f"{name}_mean"] = float(np.mean(col))
            stats[f"{name}_std"] = float(np.std(col))
            stats[f"{name}_min"] = float(np.min(col))
            stats[f"{name}_max"] = float(np.max(col))

        # Derived stats
        stats["offscreen_ratio"] = float(np.mean(matrix[:, names.index("offscreen_gaze")]))
        stats["looking_away_ratio"] = float(np.mean(matrix[:, names.index("looking_away")]))
        stats["multi_face_ratio"] = float(np.mean(matrix[:, names.index("num_faces")] > 1.0))
        stats["suspicious_object_ratio"] = float(np.mean(matrix[:, names.index("suspicious_object")]))
        stats["yaw_velocity_mean"] = float(np.mean(matrix[:, names.index("yaw_velocity")]))
        stats["gaze_velocity_mean"] = float(np.mean(matrix[:, names.index("gaze_velocity")]))

        return stats

    def get_multi_window_stats(self) -> Dict:
        """
        Hitung statistik untuk semua window sizes yang dikonfigurasi.

        Returns:
            Dictionary dengan key = window_size, value = stats
        """
        result = {}
        for ws in self.window_sizes:
            result[f"window_{ws}s"] = self.get_rolling_stats(ws)
        return result

    def get_feature_sequence(self, num_frames: int = 30) -> Optional[np.ndarray]:
        """
        Ambil sequence feature vectors terakhir untuk input ke temporal model.

        Args:
            num_frames: Jumlah frame yang diambil

        Returns:
            Array (num_frames, num_features) atau None jika belum cukup data
        """
        if len(self.history) < num_frames:
            return None

        recent = list(self.history)[-num_frames:]
        return np.array([fv.to_array() for fv in recent])

    def reset(self):
        """Reset semua history."""
        self.history.clear()
        self._prev_yaw = None
        self._prev_gaze = None
        self._prev_ts = None
