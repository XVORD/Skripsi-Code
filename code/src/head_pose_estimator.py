"""
Modul Head Pose Estimation
==========================
Memperkirakan orientasi kepala (yaw, pitch, roll) menggunakan
Perspective-n-Point (PnP) solver dari facial landmarks.

Sesuai dengan BAB 2.4 & 3.4.3 Skripsi:
- Head Pose Estimation untuk Analisis Perilaku
- Representasi Yaw, Pitch, Roll
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class HeadPoseResult:
    """Hasil estimasi pose kepala."""
    frame_number: int = 0
    timestamp: float = 0.0
    # Euler angles (derajat)
    yaw: float = 0.0      # Kiri(-) / Kanan(+)
    pitch: float = 0.0     # Atas(-) / Bawah(+)
    roll: float = 0.0      # Kemiringan kiri(-) / kanan(+)
    # Raw vectors
    rotation_vector: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    # Flags
    is_valid: bool = False
    is_suspicious_yaw: bool = False
    is_suspicious_pitch: bool = False
    is_suspicious_roll: bool = False
    is_looking_away: bool = False


class HeadPoseEstimator:
    """
    Estimasi orientasi kepala menggunakan cv2.solvePnP.

    Menggunakan 6 facial landmarks kunci yang dipetakan ke model wajah 3D
    untuk menghitung sudut yaw, pitch, dan roll.

    Fitur:
    - Real-time head pose estimation
    - Suspicious movement detection via thresholds
    - Temporal analysis: movement rate & repetitive pattern detection
    - Visualization: projected 3D axes on frame
    """

    # Model wajah 3D generik (dalam unit arbitrer, centered at nose)
    # Points: nose_tip, chin, left_eye_outer, right_eye_outer, left_mouth, right_mouth
    FACE_3D_MODEL = np.array([
        [0.0,    0.0,     0.0],        # Nose tip
        [0.0,   -330.0,  -65.0],       # Chin
        [-225.0, 170.0, -135.0],       # Left eye outer corner
        [225.0,  170.0, -135.0],       # Right eye outer corner
        [-150.0, -150.0, -125.0],      # Left mouth corner
        [150.0,  -150.0, -125.0],      # Right mouth corner
    ], dtype=np.float64)

    # Extended model keeps the first 6 points identical to FACE_3D_MODEL.
    # Extra FaceMesh points reduce pose jitter when one contour point is noisy.
    FACE_3D_MODEL_EXTENDED = np.array([
        [0.0,    0.0,     0.0],        # Nose tip
        [0.0,   -330.0,  -65.0],       # Chin
        [-225.0, 170.0, -135.0],       # Left eye outer corner
        [225.0,  170.0, -135.0],       # Right eye outer corner
        [-150.0, -150.0, -125.0],      # Left mouth corner
        [150.0,  -150.0, -125.0],      # Right mouth corner
        [-75.0,  170.0, -135.0],       # Left eye inner corner
        [75.0,   170.0, -135.0],       # Right eye inner corner
        [0.0,    330.0, -100.0],       # Forehead
        [-255.0,   0.0, -100.0],       # Left cheek
        [255.0,    0.0, -100.0],       # Right cheek
    ], dtype=np.float64)

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Inisialisasi HeadPoseEstimator.

        Args:
            config_path: Path ke file konfigurasi YAML
        """
        config = self._load_config(config_path)
        hp_config = config.get("head_pose", {})

        # Thresholds
        self.yaw_threshold = hp_config.get("yaw_threshold", 30.0)
        self.pitch_threshold = hp_config.get("pitch_threshold", 25.0)
        self.roll_threshold = hp_config.get("roll_threshold", 20.0)
        self.use_extended_landmarks = bool(hp_config.get("use_extended_landmarks", False))
        self.enable_side_profile_fallback = bool(
            hp_config.get("enable_side_profile_fallback", False)
        )
        self.side_profile_eye_offset_ratio_threshold = float(
            hp_config.get("side_profile_eye_offset_ratio_threshold", 0.55)
        )
        self.side_profile_mouth_offset_ratio_threshold = float(
            hp_config.get("side_profile_mouth_offset_ratio_threshold", 0.70)
        )

        # Repetitive movement detection
        self.movement_window = hp_config.get("movement_window_seconds", 5.0)
        self.repetitive_count_threshold = hp_config.get("repetitive_count_threshold", 3)

        # History for temporal analysis
        self.history: deque = deque(maxlen=300)  # ~20 seconds at 15fps

        # Camera matrix (akan di-set saat frame pertama)
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def reset(self) -> None:
        """Reset temporal/video-specific state before processing a new video."""
        self.history.clear()
        self._camera_matrix = None

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _get_camera_matrix(self, img_w: int, img_h: int) -> np.ndarray:
        """Buat camera matrix berdasarkan ukuran frame."""
        if self._camera_matrix is None:
            focal_length = img_w
            center = (img_w / 2, img_h / 2)
            self._camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
        return self._camera_matrix

    def _is_side_profile_from_landmarks(self, pose_landmarks_2d: np.ndarray) -> bool:
        """
        Conservative 2D fallback for clear side-profile faces.

        PnP can underestimate yaw when FaceMesh landmarks are partially occluded.
        A clear profile view has the nose tip far from the eye/mouth midline.
        """
        if not self.enable_side_profile_fallback or pose_landmarks_2d is None:
            return False
        if len(pose_landmarks_2d) < 6:
            return False

        pts = pose_landmarks_2d.astype(np.float64)
        nose_x = float(pts[0, 0])
        left_eye_x = float(pts[2, 0])
        right_eye_x = float(pts[3, 0])
        left_mouth_x = float(pts[4, 0])
        right_mouth_x = float(pts[5, 0])

        eye_span = abs(right_eye_x - left_eye_x)
        mouth_span = abs(right_mouth_x - left_mouth_x)
        if eye_span < 1.0 or mouth_span < 1.0:
            return False

        eye_mid = 0.5 * (left_eye_x + right_eye_x)
        mouth_mid = 0.5 * (left_mouth_x + right_mouth_x)
        nose_eye_offset_ratio = abs(nose_x - eye_mid) / eye_span
        nose_mouth_offset_ratio = abs(nose_x - mouth_mid) / mouth_span

        return bool(
            nose_eye_offset_ratio >= self.side_profile_eye_offset_ratio_threshold
            and nose_mouth_offset_ratio >= self.side_profile_mouth_offset_ratio_threshold
        )

    @staticmethod
    def _wrap_angle_180(angle_deg: float) -> float:
        """Normalize angle to [-180, 180)."""
        return float((angle_deg + 180.0) % 360.0 - 180.0)

    @classmethod
    def _fold_angle_90(cls, angle_deg: float) -> float:
        """
        Fold Euler angle into [-90, 90] equivalent branch.

        This mitigates decomposeProjectionMatrix branch flips where values near
        +/-180 represent a near-frontal orientation in another equivalent form.
        """
        a = cls._wrap_angle_180(angle_deg)
        if a > 90.0:
            a -= 180.0
        elif a < -90.0:
            a += 180.0
        return float(a)

    def estimate(self, pose_landmarks_2d: np.ndarray,
                 img_w: int, img_h: int,
                 frame_number: int = 0,
                 timestamp: float = 0.0) -> HeadPoseResult:
        """
        Estimasi head pose dari 6 facial landmarks.

        Args:
            pose_landmarks_2d: Array (6, 2) berisi pixel coordinates dari
                               6 key facial landmarks (nose, chin, eyes, mouth)
            img_w: Lebar frame
            img_h: Tinggi frame
            frame_number: Nomor frame
            timestamp: Timestamp (detik)

        Returns:
            HeadPoseResult
        """
        result = HeadPoseResult(frame_number=frame_number, timestamp=timestamp)

        if pose_landmarks_2d is None or len(pose_landmarks_2d) < 6:
            return result

        # Camera matrix
        cam_matrix = self._get_camera_matrix(img_w, img_h)

        # Solve PnP
        model_points = self.FACE_3D_MODEL
        image_points = pose_landmarks_2d[:6].astype(np.float64)
        if self.use_extended_landmarks and len(pose_landmarks_2d) >= len(self.FACE_3D_MODEL_EXTENDED):
            model_points = self.FACE_3D_MODEL_EXTENDED
            image_points = pose_landmarks_2d[:len(model_points)].astype(np.float64)

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            cam_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return result

        result.is_valid = True
        result.rotation_vector = rotation_vec
        result.translation_vector = translation_vec

        # Convert rotation vector ke rotation matrix, lalu ke Euler angles
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat([rotation_mat, translation_vec])  # (3, 4)
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        # Euler angles: pitch, yaw, roll
        result.pitch = float(euler_angles[0][0])
        result.yaw = float(euler_angles[1][0])
        result.roll = float(euler_angles[2][0])

        # Normalize angles.
        # - Yaw uses full [-180, 180) domain.
        # - Pitch/Roll are folded to [-90, 90] to avoid 180-degree branch flips.
        result.yaw = self._wrap_angle_180(result.yaw)
        result.pitch = self._fold_angle_90(result.pitch)
        result.roll = self._fold_angle_90(result.roll)

        # Check thresholds
        result.is_suspicious_yaw = abs(result.yaw) > self.yaw_threshold
        result.is_suspicious_pitch = abs(result.pitch) > self.pitch_threshold
        result.is_suspicious_roll = abs(result.roll) > self.roll_threshold
        if self._is_side_profile_from_landmarks(pose_landmarks_2d):
            result.is_suspicious_yaw = True
        result.is_looking_away = result.is_suspicious_yaw or result.is_suspicious_pitch

        # Store in history
        self.history.append(result)

        return result

    def get_movement_stats(self, window_seconds: Optional[float] = None) -> Dict:
        """
        Hitung statistik pergerakan dalam window waktu tertentu.

        Args:
            window_seconds: Window waktu (detik). None = gunakan config.

        Returns:
            Dictionary berisi mean, std, rate of change untuk yaw/pitch/roll
        """
        if len(self.history) < 2:
            return {"valid": False}

        window = window_seconds or self.movement_window
        current_time = self.history[-1].timestamp
        start_time = current_time - window

        # Filter history dalam window
        windowed = [h for h in self.history if h.timestamp >= start_time and h.is_valid]
        if len(windowed) < 2:
            return {"valid": False}

        yaws = [h.yaw for h in windowed]
        pitches = [h.pitch for h in windowed]
        rolls = [h.roll for h in windowed]

        # Rate of change (deg/sec)
        yaw_diffs = np.abs(np.diff(yaws))
        pitch_diffs = np.abs(np.diff(pitches))

        # Hitung berapa kali kepala berubah arah secara signifikan
        yaw_direction_changes = 0
        for i in range(1, len(yaws)):
            if abs(yaws[i] - yaws[i-1]) > 10:  # 10 degree change
                yaw_direction_changes += 1

        # Count repeated side-look events (transition into |yaw| > threshold).
        # This better matches the S2 protocol: side look repeated >= N times.
        side_look = np.abs(np.asarray(yaws, dtype=np.float64)) > float(self.yaw_threshold)
        side_look_events = 0
        prev = False
        for cur in side_look:
            if bool(cur) and not prev:
                side_look_events += 1
            prev = bool(cur)

        return {
            "valid": True,
            "window_seconds": window,
            "num_samples": len(windowed),
            "yaw_mean": float(np.mean(yaws)),
            "yaw_std": float(np.std(yaws)),
            "yaw_range": float(np.ptp(yaws)),
            "yaw_rate_mean": float(np.mean(yaw_diffs)) if len(yaw_diffs) > 0 else 0,
            "pitch_mean": float(np.mean(pitches)),
            "pitch_std": float(np.std(pitches)),
            "pitch_range": float(np.ptp(pitches)),
            "pitch_rate_mean": float(np.mean(pitch_diffs)) if len(pitch_diffs) > 0 else 0,
            "roll_mean": float(np.mean(rolls)),
            "roll_std": float(np.std(rolls)),
            "direction_changes": yaw_direction_changes,
            "side_look_events": int(side_look_events),
            "is_repetitive": side_look_events >= self.repetitive_count_threshold,
        }

    def draw_pose(self, frame_bgr: np.ndarray, result: HeadPoseResult,
                  nose_point: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Gambar visualisasi head pose pada frame.
        Menampilkan projected 3D axes dari hidung.

        Args:
            frame_bgr: Frame BGR
            result: HeadPoseResult
            nose_point: Titik hidung di pixel coordinate (opsional)

        Returns:
            Frame dengan visualisasi pose
        """
        output = frame_bgr.copy()

        if not result.is_valid:
            return output

        h, w = output.shape[:2]
        cam_matrix = self._get_camera_matrix(w, h)

        # Project 3D axes
        axis_length = 100
        axis_3d = np.array([
            [0, 0, 0],                    # Origin (nose)
            [axis_length, 0, 0],           # X-axis (merah)
            [0, axis_length, 0],           # Y-axis (hijau)
            [0, 0, axis_length],           # Z-axis (biru)
        ], dtype=np.float64)

        projected, _ = cv2.projectPoints(
            axis_3d, result.rotation_vector, result.translation_vector,
            cam_matrix, self._dist_coeffs
        )

        origin = tuple(projected[0].ravel().astype(int))
        x_end = tuple(projected[1].ravel().astype(int))
        y_end = tuple(projected[2].ravel().astype(int))
        z_end = tuple(projected[3].ravel().astype(int))

        # Draw axes
        cv2.line(output, origin, x_end, (0, 0, 255), 3)    # X = Red
        cv2.line(output, origin, y_end, (0, 255, 0), 3)    # Y = Green
        cv2.line(output, origin, z_end, (255, 0, 0), 3)    # Z = Blue

        # Angle text
        color = (0, 255, 0) if not result.is_looking_away else (0, 0, 255)
        info_y = h - 80
        cv2.putText(output, f"Yaw:   {result.yaw:+.1f}°", (10, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(output, f"Pitch: {result.pitch:+.1f}°", (10, info_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(output, f"Roll:  {result.roll:+.1f}°", (10, info_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if result.is_looking_away:
            cv2.putText(output, "LOOKING AWAY!", (w - 250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return output


# --- Quick Test ---
if __name__ == "__main__":
    from video_acquisition import VideoAcquisition
    from face_detection import FaceDetector

    print("=== Test Head Pose Estimation ===")

    va = VideoAcquisition()
    fd = FaceDetector()
    hpe = HeadPoseEstimator()

    if va.start("webcam"):
        for frame_num, original, preprocessed in va.stream_frames(max_frames=500):
            face_result = fd.detect(preprocessed, frame_num, frame_num / 15.0)

            display = original.copy()

            if face_result.face_present and len(face_result.faces) > 0:
                face = face_result.faces[0]  # Primary face
                h, w = original.shape[:2]

                pose_result = hpe.estimate(
                    face["pose_landmarks_2d"], w, h,
                    frame_num, frame_num / 15.0
                )

                # Draw face bbox + pose axes
                display = fd.draw_detection(display, face_result, draw_mesh=False)
                display = hpe.draw_pose(display, pose_result)

                # Movement stats
                stats = hpe.get_movement_stats()
                if stats.get("valid"):
                    cv2.putText(display, f"Yaw STD: {stats['yaw_std']:.1f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    if stats["is_repetitive"]:
                        cv2.putText(display, "REPETITIVE MOVEMENT!",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Head Pose Estimation Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        va.stop()
        fd.release()
        cv2.destroyAllWindows()
