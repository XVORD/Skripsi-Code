"""
Modul Eye Gaze Estimation dan Analisis Gerak Mata
=================================================
Estimasi arah pandangan mata menggunakan iris tracking dari MediaPipe,
serta deteksi fixation, saccade, dan reading patterns.

Sesuai dengan BAB 2.5 & 3.4.3 Skripsi:
- Eye Gaze Estimation dan Analisis Gerak Mata
- Konsep Gaze, Fixation, dan Saccade
- Pola Gerak Mata untuk Aktivitas Membaca dan Atensi
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class GazeResult:
    """Hasil estimasi pandangan mata."""
    frame_number: int = 0
    timestamp: float = 0.0
    # Gaze direction (normalized, -1 to 1)
    gaze_x: float = 0.0   # Kiri(-) ke Kanan(+)
    gaze_y: float = 0.0   # Atas(-) ke Bawah(+)
    # Raw iris positions (pixel)
    left_iris: Optional[Tuple[float, float]] = None
    right_iris: Optional[Tuple[float, float]] = None
    # Eye openness ratio
    left_eye_ratio: float = 0.0
    right_eye_ratio: float = 0.0
    # Flags
    is_valid: bool = False
    is_offscreen: bool = False
    is_fixating: bool = False
    is_saccade: bool = False
    is_reading_pattern: bool = False
    is_looking_at_camera: bool = False
    is_occlusion_fallback: bool = False
    direct_confidence: float = 0.0
    fallback_confidence: float = 0.0
    fusion_weight_direct: float = 1.0
    used_head_pose_fallback: bool = False
    used_lower_lid_fallback: bool = False
    used_long_blink_down_rule: bool = False
    is_short_blink: bool = False
    suppressed_by_short_blink: bool = False
    suppressed_by_frontal_guard: bool = False
    closed_eye_duration_sec: float = 0.0


class EyeGazeEstimator:
    """
    Estimasi arah pandangan mata menggunakan iris tracking.

    Menghitung posisi relatif iris dalam bounding box mata untuk
    memperkirakan ke mana kandidat melihat.

    Fitur:
    - Gaze direction estimation (x, y normalized)
    - Fixation detection (berhenti di satu titik)
    - Saccade detection (gerakan mata cepat)
    - Reading pattern detection (pola horizontal saccade berulang)
    - Off-screen gaze tracking
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        config = self._load_config(config_path)
        gaze_config = config.get("eye_gaze", {})
        video_config = config.get("video", {})

        # Fixation params
        self.fixation_threshold = gaze_config.get("fixation_threshold_pixels", 15.0)
        self.fixation_min_duration_ms = gaze_config.get("fixation_min_duration_ms", 200)

        # Saccade params
        self.saccade_velocity_threshold = gaze_config.get("saccade_velocity_threshold", 300.0)

        # Off-screen params
        self.offscreen_threshold = gaze_config.get("offscreen_gaze_threshold", 0.3)

        # Reading pattern params
        self.reading_horizontal_ratio = gaze_config.get("reading_horizontal_ratio", 2.0)
        self.reading_window_seconds = gaze_config.get("reading_window_seconds", 3.0)

        # Eye-occlusion fallback (approximation inspired by robust gaze models):
        # keep plausible gaze direction when iris signal is unreliable.
        self.enable_occlusion_fallback = bool(gaze_config.get("enable_occlusion_fallback", True))
        self.closed_eye_ratio_threshold = float(gaze_config.get("closed_eye_ratio_threshold", 0.14))
        self.occlusion_hold_seconds = float(gaze_config.get("occlusion_hold_seconds", 0.8))
        self.occlusion_downward_min_gaze_y = float(gaze_config.get("occlusion_downward_min_gaze_y", 0.38))
        self.occlusion_downward_activation_frames = int(gaze_config.get("occlusion_downward_activation_frames", 2))
        self.occlusion_x_damping = float(gaze_config.get("occlusion_x_damping", 0.4))
        # Head-pose fallback when pupil/iris is unreliable or blocked.
        self.enable_head_pose_fallback = bool(gaze_config.get("enable_head_pose_fallback", True))
        self.head_pose_pitch_down_threshold = float(gaze_config.get("head_pose_pitch_down_threshold", 20.0))
        self.head_pose_pitch_norm_deg = float(gaze_config.get("head_pose_pitch_norm_deg", 35.0))
        self.head_pose_yaw_norm_deg = float(gaze_config.get("head_pose_yaw_norm_deg", 40.0))
        self.head_pose_downward_min_gaze_y = float(gaze_config.get("head_pose_downward_min_gaze_y", 0.42))
        # Explicit rule requested for downward gaze under eyelid occlusion:
        # if eyes are closed and head is down, force gaze_y to downward.
        self.enable_closed_eye_head_down_rule = bool(
            gaze_config.get("enable_closed_eye_head_down_rule", True)
        )
        self.closed_eye_head_down_pitch_threshold = float(
            gaze_config.get("closed_eye_head_down_pitch_threshold", 12.0)
        )
        self.closed_eye_head_down_min_gaze_y = float(
            gaze_config.get("closed_eye_head_down_min_gaze_y", 0.45)
        )
        self.closed_eye_head_down_activation_frames = int(
            gaze_config.get("closed_eye_head_down_activation_frames", 1)
        )
        self.closed_eye_head_down_x_damping = float(
            gaze_config.get("closed_eye_head_down_x_damping", 0.35)
        )
        # Long eye closure rule:
        # a closure longer than a normal blink is treated as a downward glance.
        self.enable_long_blink_down_rule = bool(
            gaze_config.get("enable_long_blink_down_rule", True)
        )
        self.long_blink_duration_sec = float(gaze_config.get("long_blink_duration_sec", 0.7))
        self.long_blink_min_gaze_y = float(gaze_config.get("long_blink_min_gaze_y", 0.45))
        self.long_blink_x_damping = float(gaze_config.get("long_blink_x_damping", 0.35))
        self.long_blink_max_gap_sec = float(gaze_config.get("long_blink_max_gap_sec", 0.25))
        # Normal blink guard: short eye closures are physiological, so they
        # should not become off-screen gaze labels.
        self.enable_short_blink_suppression = bool(
            gaze_config.get("enable_short_blink_suppression", True)
        )
        self.short_blink_suppress_seconds = float(
            gaze_config.get("short_blink_suppress_seconds", 0.5)
        )
        # Borderline frontal guard:
        # suppress weak offscreen predictions when face/head is frontal and no
        # occlusion/downward evidence is active. This targets neutral-gaze bias.
        self.enable_borderline_frontal_guard = bool(
            gaze_config.get("enable_borderline_frontal_guard", True)
        )
        self.borderline_frontal_guard_margin = float(
            gaze_config.get("borderline_frontal_guard_margin", 0.07)
        )
        self.borderline_frontal_guard_yaw_abs_max = float(
            gaze_config.get("borderline_frontal_guard_yaw_abs_max", 12.0)
        )
        self.borderline_frontal_guard_pitch_abs_max = float(
            gaze_config.get("borderline_frontal_guard_pitch_abs_max", 10.0)
        )
        self.borderline_frontal_guard_gaze_y_abs_max = float(
            gaze_config.get("borderline_frontal_guard_gaze_y_abs_max", 0.28)
        )
        self.borderline_frontal_guard_min_eye_ratio = float(
            gaze_config.get("borderline_frontal_guard_min_eye_ratio", 0.16)
        )
        # Partial eyelid occlusion rule:
        # detects downward gaze that looks like a blink because the pupil/iris is
        # partly covered by the eyelid while the eye is still not fully closed.
        self.enable_lower_lid_occlusion_rule = bool(
            gaze_config.get("enable_lower_lid_occlusion_rule", True)
        )
        self.lower_lid_eye_ratio_min = float(gaze_config.get("lower_lid_eye_ratio_min", 0.12))
        self.lower_lid_eye_ratio_max = float(gaze_config.get("lower_lid_eye_ratio_max", 0.27))
        self.lower_lid_gaze_y_threshold = float(
            gaze_config.get("lower_lid_gaze_y_threshold", 0.18)
        )
        self.lower_lid_pitch_down_threshold = float(
            gaze_config.get("lower_lid_pitch_down_threshold", 12.0)
        )
        self.lower_lid_pitch_gaze_y_min = float(
            gaze_config.get("lower_lid_pitch_gaze_y_min", -0.06)
        )
        self.lower_lid_direct_confidence_max = float(
            gaze_config.get("lower_lid_direct_confidence_max", 0.82)
        )
        self.lower_lid_min_gaze_y = float(gaze_config.get("lower_lid_min_gaze_y", 0.36))
        self.lower_lid_activation_frames = int(
            gaze_config.get("lower_lid_activation_frames", 1)
        )
        self.lower_lid_x_damping = float(gaze_config.get("lower_lid_x_damping", 0.35))
        # Open-eye frontal downward gaze rule:
        # sustained eye-only downward gaze maps into the existing offscreen flag.
        self.enable_frontal_downward_gaze_rule = bool(
            gaze_config.get("enable_frontal_downward_gaze_rule", False)
        )
        self.frontal_downward_gaze_y_threshold = float(
            gaze_config.get("frontal_downward_gaze_y_threshold", 0.18)
        )
        self.frontal_downward_gaze_abs_x_max = float(
            gaze_config.get("frontal_downward_gaze_abs_x_max", 0.25)
        )
        self.frontal_downward_gaze_yaw_abs_max = float(
            gaze_config.get("frontal_downward_gaze_yaw_abs_max", 12.0)
        )
        self.frontal_downward_gaze_pitch_abs_max = float(
            gaze_config.get("frontal_downward_gaze_pitch_abs_max", 12.0)
        )
        self.frontal_downward_gaze_min_eye_ratio = float(
            gaze_config.get("frontal_downward_gaze_min_eye_ratio", 0.15)
        )
        self.frontal_downward_gaze_activation_frames = int(
            gaze_config.get("frontal_downward_gaze_activation_frames", 6)
        )
        # Confidence-weighted fusion
        self.enable_confidence_fusion = bool(gaze_config.get("enable_confidence_fusion", True))
        # Conservative defaults: preserve direct iris estimate unless occlusion
        # evidence is strong, to avoid FP inflation.
        self.direct_confidence_floor = float(gaze_config.get("direct_confidence_floor", 0.45))
        self.open_eye_ratio_reference = float(gaze_config.get("open_eye_ratio_reference", 0.26))
        self.max_lr_disagreement = float(gaze_config.get("max_lr_disagreement", 0.7))
        self.fusion_min_direct_weight = float(gaze_config.get("fusion_min_direct_weight", 0.45))

        self.fps = video_config.get("fps", 15)

        # History
        self.history: deque = deque(maxlen=300)

        # Fixation tracking
        self._fixation_start_time: Optional[float] = None
        self._fixation_center: Optional[Tuple[float, float]] = None
        self._current_fixation_duration: float = 0.0
        self._occlusion_run_frames: int = 0
        self._lower_lid_run_frames: int = 0
        self._frontal_downward_gaze_run_frames: int = 0
        self._closed_eye_start_ts: Optional[float] = None
        self._last_closed_eye_ts: Optional[float] = None

    def reset(self) -> None:
        """Reset temporal state before processing a new video."""
        self.history.clear()
        self._fixation_start_time = None
        self._fixation_center = None
        self._current_fixation_duration = 0.0
        self._occlusion_run_frames = 0
        self._lower_lid_run_frames = 0
        self._frontal_downward_gaze_run_frames = 0
        self._closed_eye_start_ts = None
        self._last_closed_eye_ts = None

    def _last_valid_gaze(self, max_age_seconds: float) -> Optional[GazeResult]:
        if not self.history:
            return None
        current_ts = self.history[-1].timestamp
        for h in reversed(self.history):
            if not h.is_valid:
                continue
            if (current_ts - h.timestamp) <= max_age_seconds:
                return h
            break
        return None

    def _head_pose_prior(self, head_pose_result: Optional[object]) -> Optional[Tuple[float, float]]:
        if not self.enable_head_pose_fallback or head_pose_result is None:
            return None
        is_valid = getattr(head_pose_result, "is_valid", True)
        if not is_valid:
            return None
        try:
            yaw = float(getattr(head_pose_result, "yaw"))
            pitch = float(getattr(head_pose_result, "pitch"))
        except Exception:
            return None

        gx = float(np.clip(yaw / max(1e-6, self.head_pose_yaw_norm_deg), -1.0, 1.0))
        gy = float(np.clip(pitch / max(1e-6, self.head_pose_pitch_norm_deg), -1.0, 1.0))
        if pitch >= self.head_pose_pitch_down_threshold:
            gy = max(gy, float(self.head_pose_downward_min_gaze_y))
        return gx, gy

    def _apply_occlusion_fallback(self, result: GazeResult, eye_closed: bool,
                                  head_pose_result: Optional[object] = None) -> bool:
        if not self.enable_occlusion_fallback:
            return False
        prev = self._last_valid_gaze(self.occlusion_hold_seconds)
        pose_prior = self._head_pose_prior(head_pose_result)
        if prev is None and pose_prior is None:
            return False

        # Carry last reliable gaze. For closed eyes, bias to downward
        # to preserve "looking down" behavior that often gets lost on blinks.
        result.is_valid = True
        result.is_occlusion_fallback = True
        result.fallback_confidence = 1.0 if prev is not None else 0.8
        result.fusion_weight_direct = 0.0
        if prev is not None:
            base_x = float(prev.gaze_x) * float(self.occlusion_x_damping)
            down_y = float(prev.gaze_y)
        else:
            base_x = 0.0
            down_y = 0.0
        if eye_closed and self._occlusion_run_frames >= self.occlusion_downward_activation_frames:
            down_y = max(down_y, float(self.occlusion_downward_min_gaze_y))
        if pose_prior is not None:
            pose_x, pose_y = pose_prior
            # More trust to pose on Y (down/up), milder trust on X.
            base_x = 0.6 * base_x + 0.4 * pose_x
            down_y = 0.4 * down_y + 0.6 * pose_y
            result.used_head_pose_fallback = True
        result.gaze_x = float(np.clip(base_x, -1.0, 1.0))
        result.gaze_y = float(np.clip(down_y, -1.0, 1.0))
        return True

    def _apply_closed_eye_head_down_rule(self, result: GazeResult, eye_closed: bool,
                                         head_pose_result: Optional[object] = None):
        """
        Force downward gaze when:
        - eye is closed (eyelid occlusion), and
        - head pitch indicates looking down.
        """
        if not self.enable_closed_eye_head_down_rule:
            return
        if not eye_closed:
            return
        if self._occlusion_run_frames < self.closed_eye_head_down_activation_frames:
            return
        if head_pose_result is None or not getattr(head_pose_result, "is_valid", False):
            return
        try:
            pitch = float(getattr(head_pose_result, "pitch"))
        except Exception:
            return
        if pitch < self.closed_eye_head_down_pitch_threshold:
            return

        # Convert pitch to a bounded downward prior [0, 1], then enforce minimum.
        pitch_based_down = float(np.clip(pitch / max(1e-6, self.head_pose_pitch_norm_deg), 0.0, 1.0))
        enforced_y = max(
            result.gaze_y,
            self.occlusion_downward_min_gaze_y,
            self.head_pose_downward_min_gaze_y,
            self.closed_eye_head_down_min_gaze_y,
            pitch_based_down,
        )
        result.is_valid = True
        result.gaze_y = float(np.clip(enforced_y, -1.0, 1.0))
        result.gaze_x = float(np.clip(result.gaze_x * self.closed_eye_head_down_x_damping, -1.0, 1.0))
        result.is_occlusion_fallback = True
        result.used_head_pose_fallback = True

    def _update_eye_closure_duration(self, eye_closed: bool, timestamp: float) -> float:
        if not eye_closed:
            self._closed_eye_start_ts = None
            self._last_closed_eye_ts = None
            return 0.0

        ts = float(timestamp)
        if (
            self._closed_eye_start_ts is None
            or self._last_closed_eye_ts is None
            or (ts - self._last_closed_eye_ts) > self.long_blink_max_gap_sec
        ):
            self._closed_eye_start_ts = ts

        self._last_closed_eye_ts = ts
        return max(0.0, ts - float(self._closed_eye_start_ts))

    def _apply_long_blink_down_rule(self, result: GazeResult, eye_closed: bool):
        if not self.enable_long_blink_down_rule:
            return
        if not eye_closed:
            return
        if result.closed_eye_duration_sec < self.long_blink_duration_sec:
            return

        result.is_valid = True
        result.gaze_y = float(np.clip(max(result.gaze_y, self.long_blink_min_gaze_y), -1.0, 1.0))
        result.gaze_x = float(np.clip(result.gaze_x * self.long_blink_x_damping, -1.0, 1.0))
        result.is_occlusion_fallback = True
        result.used_long_blink_down_rule = True

    def _should_suppress_borderline_frontal(self,
                                            result: GazeResult,
                                            gaze_distance: float,
                                            mean_eye_ratio: float,
                                            head_pose_result: Optional[object]) -> bool:
        if not self.enable_borderline_frontal_guard:
            return False
        if not result.is_valid:
            return False
        if gaze_distance <= self.offscreen_threshold:
            return False
        if gaze_distance > self.offscreen_threshold + self.borderline_frontal_guard_margin:
            return False
        if (
            result.is_occlusion_fallback
            or result.used_head_pose_fallback
            or result.used_lower_lid_fallback
            or result.used_long_blink_down_rule
        ):
            return False
        if mean_eye_ratio < self.borderline_frontal_guard_min_eye_ratio:
            return False
        if abs(result.gaze_y) > self.borderline_frontal_guard_gaze_y_abs_max:
            return False
        if head_pose_result is None or not getattr(head_pose_result, "is_valid", False):
            return False
        try:
            yaw = abs(float(getattr(head_pose_result, "yaw")))
            pitch = abs(float(getattr(head_pose_result, "pitch")))
        except Exception:
            return False
        return bool(
            yaw <= self.borderline_frontal_guard_yaw_abs_max
            and pitch <= self.borderline_frontal_guard_pitch_abs_max
        )

    def _get_head_pitch(self, head_pose_result: Optional[object]) -> Optional[float]:
        if head_pose_result is None or not getattr(head_pose_result, "is_valid", False):
            return None
        try:
            return float(getattr(head_pose_result, "pitch"))
        except Exception:
            return None

    def _is_lower_lid_down_candidate(self,
                                     result: GazeResult,
                                     mean_eye_ratio: float,
                                     eye_closed: bool,
                                     head_pose_result: Optional[object] = None) -> bool:
        """
        Candidate for downward gaze hidden by partial eyelid occlusion.

        This is intentionally stricter than normal blink handling:
        - eye must be partially narrowed, not fully closed,
        - and there must be downward evidence from gaze_y or head pitch.
        """
        if not self.enable_lower_lid_occlusion_rule:
            return False
        if not result.is_valid or eye_closed:
            return False
        if mean_eye_ratio <= 0:
            return False
        if mean_eye_ratio < self.lower_lid_eye_ratio_min:
            return False
        if mean_eye_ratio > self.lower_lid_eye_ratio_max:
            return False

        pitch = self._get_head_pitch(head_pose_result)
        gaze_down = result.gaze_y >= self.lower_lid_gaze_y_threshold
        pitch_down = pitch is not None and pitch >= self.lower_lid_pitch_down_threshold
        pitch_supported_down = pitch_down and result.gaze_y >= self.lower_lid_pitch_gaze_y_min
        low_conf_pose_down = (
            pitch_down
            and result.direct_confidence <= self.lower_lid_direct_confidence_max
            and result.gaze_y >= -self.lower_lid_pitch_gaze_y_min
        )
        return bool(gaze_down or pitch_supported_down or low_conf_pose_down)

    def _apply_lower_lid_occlusion_rule(self,
                                        result: GazeResult,
                                        head_pose_result: Optional[object] = None):
        if not self.enable_lower_lid_occlusion_rule:
            return
        if self._lower_lid_run_frames < self.lower_lid_activation_frames:
            return

        pitch = self._get_head_pitch(head_pose_result)
        enforced_y = max(result.gaze_y, self.lower_lid_min_gaze_y)
        if pitch is not None and pitch >= self.lower_lid_pitch_down_threshold:
            pitch_based_down = float(
                np.clip(pitch / max(1e-6, self.head_pose_pitch_norm_deg), 0.0, 1.0)
            )
            enforced_y = max(enforced_y, pitch_based_down)
            result.used_head_pose_fallback = True

        result.is_valid = True
        result.gaze_y = float(np.clip(enforced_y, -1.0, 1.0))
        result.gaze_x = float(np.clip(result.gaze_x * self.lower_lid_x_damping, -1.0, 1.0))
        result.is_occlusion_fallback = True
        result.used_lower_lid_fallback = True

    def _is_frontal_downward_gaze_candidate(self,
                                            result: GazeResult,
                                            mean_eye_ratio: float,
                                            eye_closed: bool,
                                            head_pose_result: Optional[object] = None) -> bool:
        """
        Candidate for open-eye downward gaze while the head remains frontal.

        This intentionally reuses the existing offscreen output instead of adding
        a new behavior feature.
        """
        if not self.enable_frontal_downward_gaze_rule:
            return False
        if not result.is_valid or eye_closed:
            return False
        if result.is_occlusion_fallback or result.used_head_pose_fallback:
            return False
        if result.used_lower_lid_fallback or result.used_long_blink_down_rule:
            return False
        if mean_eye_ratio < self.frontal_downward_gaze_min_eye_ratio:
            return False
        if result.gaze_y < self.frontal_downward_gaze_y_threshold:
            return False
        if abs(result.gaze_x) > self.frontal_downward_gaze_abs_x_max:
            return False
        if head_pose_result is None or not getattr(head_pose_result, "is_valid", False):
            return False
        try:
            yaw = abs(float(getattr(head_pose_result, "yaw")))
            pitch = abs(float(getattr(head_pose_result, "pitch")))
        except Exception:
            return False
        return bool(
            yaw <= self.frontal_downward_gaze_yaw_abs_max
            and pitch <= self.frontal_downward_gaze_pitch_abs_max
        )

    def _compute_direct_confidence(self,
                                   left_gaze: Tuple[float, float],
                                   right_gaze: Tuple[float, float],
                                   mean_eye_ratio: float) -> float:
        # Consistency between left/right eye gaze estimation.
        dx = float(left_gaze[0] - right_gaze[0])
        dy = float(left_gaze[1] - right_gaze[1])
        disagreement = float(np.sqrt(dx * dx + dy * dy))
        consistency = 1.0 - min(1.0, disagreement / max(1e-6, self.max_lr_disagreement))
        # Openness proxy confidence from EAR.
        openness = min(1.0, max(0.0, mean_eye_ratio / max(1e-6, self.open_eye_ratio_reference)))
        conf = 0.55 * consistency + 0.45 * openness
        return float(min(1.0, max(0.0, conf)))

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _compute_iris_position_in_eye(self, iris_center: List[float],
                                       eye_contour: np.ndarray) -> Tuple[float, float]:
        """
        Hitung posisi relatif iris dalam bounding box mata.
        Menghasilkan nilai normalized (-1 ke 1) untuk x dan y.

        Args:
            iris_center: [x, y] pixel coordinate iris center
            eye_contour: Array (N, 2) eye contour points

        Returns:
            (gaze_x, gaze_y) normalized
        """
        eye_x = eye_contour[:, 0]
        eye_y = eye_contour[:, 1]

        eye_x_min = np.min(eye_x)
        eye_x_max = np.max(eye_x)
        eye_y_min = np.min(eye_y)
        eye_y_max = np.max(eye_y)

        eye_w = eye_x_max - eye_x_min
        eye_h = eye_y_max - eye_y_min

        if eye_w == 0 or eye_h == 0:
            return 0.0, 0.0

        # Normalize iris position relative to eye bounding box
        # 0.5 = center, 0 = left/top, 1 = right/bottom
        rel_x = (iris_center[0] - eye_x_min) / eye_w
        rel_y = (iris_center[1] - eye_y_min) / eye_h

        # Convert to -1 to 1 range (0 = center)
        gaze_x = (rel_x - 0.5) * 2.0
        gaze_y = (rel_y - 0.5) * 2.0

        return float(gaze_x), float(gaze_y)

    def _compute_eye_aspect_ratio(self, eye_contour: np.ndarray) -> float:
        """
        Hitung Eye Aspect Ratio (EAR) untuk mendeteksi mata terbuka/tertutup.

        Returns:
            Rasio tinggi/lebar mata
        """
        eye_y = eye_contour[:, 1]
        eye_x = eye_contour[:, 0]

        height = np.max(eye_y) - np.min(eye_y)
        width = np.max(eye_x) - np.min(eye_x)

        if width == 0:
            return 0.0
        return float(height / width)

    def estimate(self, face_data: Dict, frame_number: int = 0,
                 timestamp: float = 0.0,
                 head_pose_result: Optional[object] = None) -> GazeResult:
        """
        Estimasi arah pandangan mata dari face detection data.

        Args:
            face_data: Dictionary dari FaceDetector.detect() — satu wajah
            frame_number: Nomor frame
            timestamp: Timestamp (detik)

        Returns:
            GazeResult
        """
        result = GazeResult(frame_number=frame_number, timestamp=timestamp)

        left_iris = face_data.get("left_iris_center")
        right_iris = face_data.get("right_iris_center")
        left_eye_contour = face_data.get("left_eye_contour")
        right_eye_contour = face_data.get("right_eye_contour")

        # Eye aspect ratio can still be available even when iris is unstable.
        if left_eye_contour is not None:
            result.left_eye_ratio = self._compute_eye_aspect_ratio(left_eye_contour)
        if right_eye_contour is not None:
            result.right_eye_ratio = self._compute_eye_aspect_ratio(right_eye_contour)
        mean_ratio = float((result.left_eye_ratio + result.right_eye_ratio) / 2.0)
        eye_closed = bool(mean_ratio > 0 and mean_ratio <= self.closed_eye_ratio_threshold)
        if eye_closed:
            self._occlusion_run_frames += 1
        else:
            self._occlusion_run_frames = 0
        result.closed_eye_duration_sec = self._update_eye_closure_duration(
            eye_closed=eye_closed,
            timestamp=timestamp,
        )
        result.is_short_blink = bool(
            eye_closed
            and self.enable_short_blink_suppression
            and result.closed_eye_duration_sec < self.short_blink_suppress_seconds
        )

        # Default path: direct iris-based gaze.
        can_direct = (
            left_iris is not None and right_iris is not None
            and left_eye_contour is not None and right_eye_contour is not None
        )
        if can_direct:
            result.is_valid = True
            result.left_iris = tuple(left_iris)
            result.right_iris = tuple(right_iris)

            left_gaze_x, left_gaze_y = self._compute_iris_position_in_eye(
                left_iris, left_eye_contour
            )
            right_gaze_x, right_gaze_y = self._compute_iris_position_in_eye(
                right_iris, right_eye_contour
            )

            # Average kedua mata
            direct_gaze_x = (left_gaze_x + right_gaze_x) / 2.0
            direct_gaze_y = (left_gaze_y + right_gaze_y) / 2.0
            result.gaze_x = float(direct_gaze_x)
            result.gaze_y = float(direct_gaze_y)
            result.direct_confidence = self._compute_direct_confidence(
                (left_gaze_x, left_gaze_y),
                (right_gaze_x, right_gaze_y),
                mean_ratio
            )
        else:
            # Fallback when iris landmarks are missing/noisy.
            if not self._apply_occlusion_fallback(
                result, eye_closed=eye_closed, head_pose_result=head_pose_result
            ):
                self._frontal_downward_gaze_run_frames = 0
                return result

        # Confidence-weighted fusion: when eyes are occluded/uncertain, blend direct gaze
        # with temporal fallback prior instead of fully trusting either source.
        if result.is_valid and self.enable_confidence_fusion:
            prev = self._last_valid_gaze(self.occlusion_hold_seconds)
            if prev is not None and can_direct:
                fallback_x = float(prev.gaze_x) * float(self.occlusion_x_damping)
                fallback_y = float(prev.gaze_y)
                if eye_closed and self._occlusion_run_frames >= self.occlusion_downward_activation_frames:
                    fallback_y = max(fallback_y, float(self.occlusion_downward_min_gaze_y))
                fallback_y = float(np.clip(fallback_y, -1.0, 1.0))
                result.fallback_confidence = 1.0

                w_direct = float(result.direct_confidence)
                if eye_closed:
                    w_direct = min(
                        w_direct,
                        max(self.fusion_min_direct_weight, self.direct_confidence_floor)
                    )
                w_direct = float(min(1.0, max(0.0, w_direct)))
                w_fallback = 1.0 - w_direct

                result.gaze_x = float(w_direct * result.gaze_x + w_fallback * fallback_x)
                result.gaze_y = float(w_direct * result.gaze_y + w_fallback * fallback_y)
                result.fusion_weight_direct = w_direct
                if eye_closed and w_fallback > 0.5:
                    result.is_occlusion_fallback = True
            elif can_direct and head_pose_result is not None:
                # If we have direct gaze but no temporal fallback, still allow pose-guided
                # stabilization on strong down-looking with closed eyes.
                pose_prior = self._head_pose_prior(head_pose_result)
                if pose_prior is not None and eye_closed:
                    px, py = pose_prior
                    w_direct = float(max(self.fusion_min_direct_weight, result.direct_confidence))
                    w_direct = float(min(1.0, max(0.0, w_direct)))
                    w_pose = 1.0 - w_direct
                    result.gaze_x = float(w_direct * result.gaze_x + w_pose * px)
                    result.gaze_y = float(w_direct * result.gaze_y + w_pose * py)
                    result.fusion_weight_direct = w_direct
                    result.used_head_pose_fallback = True

        # Assist direct prediction on prolonged eye closure.
        if result.is_valid and eye_closed and not self.enable_confidence_fusion:
            prev = self._last_valid_gaze(self.occlusion_hold_seconds)
            prev_down = (prev is not None and prev.gaze_y > 0.15)
            curr_down = result.gaze_y > 0.15
            if self._occlusion_run_frames >= self.occlusion_downward_activation_frames and (prev_down or curr_down):
                result.gaze_y = float(max(result.gaze_y, self.occlusion_downward_min_gaze_y))
                result.gaze_x = float(result.gaze_x * self.occlusion_x_damping)
                result.is_occlusion_fallback = True

        lower_lid_candidate = self._is_lower_lid_down_candidate(
            result=result,
            mean_eye_ratio=mean_ratio,
            eye_closed=eye_closed,
            head_pose_result=head_pose_result,
        )
        if lower_lid_candidate:
            self._lower_lid_run_frames += 1
        else:
            self._lower_lid_run_frames = 0

        # Partial eyelid + downward evidence => treat as downward/offscreen gaze.
        self._apply_lower_lid_occlusion_rule(
            result=result,
            head_pose_result=head_pose_result,
        )

        # Eye closed longer than a normal blink => treat as downward gaze.
        self._apply_long_blink_down_rule(
            result=result,
            eye_closed=eye_closed,
        )

        # Final explicit override for "eyes closed + head down => looking down".
        self._apply_closed_eye_head_down_rule(
            result=result,
            eye_closed=eye_closed,
            head_pose_result=head_pose_result,
        )

        frontal_downward_candidate = self._is_frontal_downward_gaze_candidate(
            result=result,
            mean_eye_ratio=mean_ratio,
            eye_closed=eye_closed,
            head_pose_result=head_pose_result,
        )
        if frontal_downward_candidate:
            self._frontal_downward_gaze_run_frames += 1
        else:
            self._frontal_downward_gaze_run_frames = 0
        frontal_downward_offscreen = (
            self._frontal_downward_gaze_run_frames
            >= self.frontal_downward_gaze_activation_frames
        )

        # Off-screen detection
        gaze_distance = np.sqrt(result.gaze_x**2 + result.gaze_y**2)
        result.is_offscreen = (
            gaze_distance > self.offscreen_threshold
            or frontal_downward_offscreen
        )
        if result.is_short_blink and result.is_offscreen:
            result.is_offscreen = False
            result.suppressed_by_short_blink = True

        if (
            result.is_offscreen
            and not frontal_downward_offscreen
            and self._should_suppress_borderline_frontal(
                result=result,
                gaze_distance=float(gaze_distance),
                mean_eye_ratio=mean_ratio,
                head_pose_result=head_pose_result,
            )
        ):
            result.is_offscreen = False
            result.suppressed_by_frontal_guard = True

        # Looking at camera (pandangan relatif lurus ke depan)
        result.is_looking_at_camera = gaze_distance < 0.15

        # Fixation & Saccade detection
        self._detect_fixation_saccade(result)

        # Reading pattern detection
        self._detect_reading_pattern(result)

        # Store in history
        self.history.append(result)

        return result

    def _detect_fixation_saccade(self, result: GazeResult):
        """Deteksi apakah mata sedang fixation atau saccade."""
        if len(self.history) == 0:
            return

        prev = self.history[-1]
        if not prev.is_valid:
            return

        # Velocity (perubahan gaze per frame)
        dx = result.gaze_x - prev.gaze_x
        dy = result.gaze_y - prev.gaze_y
        dt = result.timestamp - prev.timestamp
        if dt <= 0:
            dt = 1.0 / self.fps

        velocity = np.sqrt(dx**2 + dy**2) / dt

        # Saccade: gerakan cepat
        if velocity > (self.saccade_velocity_threshold / 1000.0):  # Convert to normalized units
            result.is_saccade = True
            self._fixation_start_time = None
            return

        # Fixation: gaze stabil di satu titik
        if self._fixation_start_time is None:
            self._fixation_start_time = result.timestamp
            self._fixation_center = (result.gaze_x, result.gaze_y)

        # Check if still within fixation area
        if self._fixation_center:
            dist = np.sqrt(
                (result.gaze_x - self._fixation_center[0])**2 +
                (result.gaze_y - self._fixation_center[1])**2
            )
            # Normalized threshold
            norm_threshold = self.fixation_threshold / 640.0  # approximate normalization
            if dist > norm_threshold:
                # Moved away — end fixation
                self._fixation_start_time = result.timestamp
                self._fixation_center = (result.gaze_x, result.gaze_y)
            else:
                # Still fixating
                duration_ms = (result.timestamp - self._fixation_start_time) * 1000
                if duration_ms >= self.fixation_min_duration_ms:
                    result.is_fixating = True
                    self._current_fixation_duration = duration_ms

    def _detect_reading_pattern(self, result: GazeResult):
        """
        Deteksi pola membaca: horizontal fixation-saccade berulang.

        Tanda membaca teks:
        - Gerakan mata dominan horizontal (kiri ke kanan)
        - Pola saccade pendek berulang
        - Fixation singkat berurutan
        """
        if len(self.history) < 10:
            return

        current_time = result.timestamp
        window_start = current_time - self.reading_window_seconds

        # Filter history dalam window
        windowed = [h for h in self.history
                    if h.timestamp >= window_start and h.is_valid]
        if len(windowed) < 5:
            return

        # Hitung rasio gerakan horizontal vs vertikal
        gaze_xs = [h.gaze_x for h in windowed]
        gaze_ys = [h.gaze_y for h in windowed]

        x_diffs = np.abs(np.diff(gaze_xs))
        y_diffs = np.abs(np.diff(gaze_ys))

        total_x_movement = np.sum(x_diffs)
        total_y_movement = np.sum(y_diffs)

        if total_y_movement == 0:
            ratio = float('inf')
        else:
            ratio = total_x_movement / total_y_movement

        # Check for reading pattern: horizontal movement dominan
        if ratio > self.reading_horizontal_ratio:
            # Additional check: saccade count
            saccade_count = sum(1 for h in windowed if h.is_saccade)
            if saccade_count >= 3:
                result.is_reading_pattern = True

    def get_gaze_stats(self, window_seconds: float = 5.0) -> Dict:
        """
        Statistik gaze dalam window waktu tertentu.

        Returns:
            Dictionary berisi statistik gaze
        """
        if len(self.history) < 2:
            return {"valid": False}

        current_time = self.history[-1].timestamp
        start_time = current_time - window_seconds

        windowed = [h for h in self.history
                    if h.timestamp >= start_time and h.is_valid]
        if len(windowed) < 2:
            return {"valid": False}

        gaze_xs = [h.gaze_x for h in windowed]
        gaze_ys = [h.gaze_y for h in windowed]

        # Off-screen analysis
        offscreen_count = sum(1 for h in windowed if h.is_offscreen)
        offscreen_ratio = offscreen_count / len(windowed)

        # Fixation analysis
        fixation_count = sum(1 for h in windowed if h.is_fixating)
        saccade_count = sum(1 for h in windowed if h.is_saccade)
        reading_count = sum(1 for h in windowed if h.is_reading_pattern)

        return {
            "valid": True,
            "window_seconds": window_seconds,
            "num_samples": len(windowed),
            "gaze_x_mean": float(np.mean(gaze_xs)),
            "gaze_x_std": float(np.std(gaze_xs)),
            "gaze_y_mean": float(np.mean(gaze_ys)),
            "gaze_y_std": float(np.std(gaze_ys)),
            "offscreen_ratio": offscreen_ratio,
            "fixation_count": fixation_count,
            "saccade_count": saccade_count,
            "reading_pattern_count": reading_count,
            "is_mostly_offscreen": offscreen_ratio > 0.5,
        }

    def draw_gaze(self, frame_bgr: np.ndarray, result: GazeResult) -> np.ndarray:
        """
        Gambar visualisasi arah pandangan pada frame.
        """
        output = frame_bgr.copy()
        h, w = output.shape[:2]

        if not result.is_valid:
            return output

        # Gaze point visualization (titik di layar yang dilihat)
        gaze_screen_x = int(w / 2 + result.gaze_x * w / 2)
        gaze_screen_y = int(h / 2 + result.gaze_y * h / 2)
        gaze_screen_x = np.clip(gaze_screen_x, 0, w - 1)
        gaze_screen_y = np.clip(gaze_screen_y, 0, h - 1)

        # Cross-hair at gaze point
        color = (0, 255, 0)
        if result.is_offscreen:
            color = (0, 0, 255)
        elif result.is_reading_pattern:
            color = (0, 165, 255)  # Orange

        cv2.drawMarker(output, (gaze_screen_x, gaze_screen_y),
                       color, cv2.MARKER_CROSS, 20, 2)

        # Info text
        info_x = w - 280
        cv2.putText(output, f"Gaze X: {result.gaze_x:+.2f}", (info_x, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output, f"Gaze Y: {result.gaze_y:+.2f}", (info_x, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Status indicators
        status_y = h - 30
        if result.is_fixating:
            cv2.putText(output, "FIXATING", (info_x, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        if result.is_saccade:
            cv2.putText(output, "SACCADE", (info_x, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        if result.is_reading_pattern:
            cv2.putText(output, "READING DETECTED!", (10, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        if result.is_offscreen:
            cv2.putText(output, "OFF-SCREEN GAZE!", (10, h - 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return output


# --- Quick Test ---
if __name__ == "__main__":
    from video_acquisition import VideoAcquisition
    from face_detection import FaceDetector

    print("=== Test Eye Gaze Estimation ===")

    va = VideoAcquisition()
    fd = FaceDetector()
    ege = EyeGazeEstimator()

    if va.start("webcam"):
        for frame_num, original, preprocessed in va.stream_frames(max_frames=500):
            face_result = fd.detect(preprocessed, frame_num, frame_num / 15.0)

            display = original.copy()

            if face_result.face_present and len(face_result.faces) > 0:
                face = face_result.faces[0]
                gaze_result = ege.estimate(face, frame_num, frame_num / 15.0)
                display = fd.draw_detection(display, face_result, draw_mesh=False)
                display = ege.draw_gaze(display, gaze_result)

            cv2.imshow("Eye Gaze Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        va.stop()
        fd.release()
        cv2.destroyAllWindows()
