"""
Modul Penilaian Kecurangan
==========================
Menggabungkan hasil analisis visual dan temporal menjadi skor
kecurangan akhir dengan pembobotan per indikator.

Sesuai dengan BAB 3.5 Skripsi:
- Indikator Perilaku Mencurigakan
- Pembobotan Skor Kecurangan
- Perhitungan Skor Akhir
"""

import numpy as np
import yaml
import time
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class RiskLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    SUSPICIOUS = "suspicious"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class CheatingScore:
    """Skor kecurangan untuk satu interval waktu."""
    timestamp: float = 0.0
    interval_start: float = 0.0
    interval_end: float = 0.0

    # Per-indicator scores (0.0 to 1.0)
    indicator_scores: Dict[str, float] = field(default_factory=dict)
    normalized_indicator_scores: Dict[str, float] = field(default_factory=dict)
    effective_weights: Dict[str, float] = field(default_factory=dict)

    # Weighted combined score
    total_score: float = 0.0
    score_percent: float = 0.0

    # Risk level
    risk_level: RiskLevel = RiskLevel.NORMAL

    # Temporal model prediction
    temporal_prediction: str = "normal"
    temporal_confidence: float = 0.0

    # Breakdown
    top_indicators: List[str] = field(default_factory=list)

    # Cheating type probabilities
    prob_third_party: float = 0.0
    prob_ai_assistance: float = 0.0


class CheatingScorer:
    """
    Sistem penilaian kecurangan berbasis weighted scoring.

    Mengumpulkan data dari semua modul analisis dan menghitung
    skor kecurangan per interval waktu.
    """

    INDICATOR_NAMES = [
        "face_absent",
        "multiple_faces",
        "offscreen_gaze",
        "suspicious_object",
        "reading_pattern",
        "head_movement_anomaly",
        "response_latency",
        "temporal_anomaly",
    ]

    def __init__(self, config_path: str = "configs/config.yaml"):
        config = self._load_config(config_path)
        score_config = config.get("scoring", {})

        # Weights per indicator
        self.weights = score_config.get("weights", {
            "face_absent": 0.15,
            "multiple_faces": 0.15,
            "offscreen_gaze": 0.15,
            "suspicious_object": 0.15,
            "reading_pattern": 0.15,
            "head_movement_anomaly": 0.10,
            "response_latency": 0.10,
            "temporal_anomaly": 0.05,
        })
        self.dynamic_weighting = bool(score_config.get("dynamic_weighting", True))
        self.face_absent_consecutive_seconds = float(
            score_config.get("face_absent_consecutive_seconds", 3.0)
        )
        # Risk scheme:
        # - legacy4  : normal/suspicious/high_risk/critical
        # - thesis3  : normal/warning/suspicious
        self.risk_scheme = str(score_config.get("risk_scheme", "legacy4")).lower()

        # Risk thresholds
        thresholds = score_config.get("thresholds", {})
        self.threshold_warning = float(thresholds.get("warning", 0.3))
        self.threshold_suspicious = float(thresholds.get("suspicious", 0.7))
        self.threshold_high_risk = float(thresholds.get("high_risk", 0.6))
        self.threshold_critical = float(thresholds.get("critical", 0.8))
        if self.risk_scheme == "legacy4":
            # Backward-compatible defaults for old config/test suite.
            self.threshold_suspicious = float(thresholds.get("suspicious", 0.3))

        # Optional sigmoid mapping (alignable with formula 3.11)
        self.use_sigmoid = bool(score_config.get("use_sigmoid", False))
        self.sigmoid_scale = float(score_config.get("sigmoid_scale", 1.0))
        self.sigmoid_bias = float(score_config.get("sigmoid_bias", 0.0))

        # Scoring interval
        self.interval_seconds = score_config.get("interval_seconds", 10.0)

        # History
        self.score_history: deque = deque(maxlen=100)
        self._current_interval_start: Optional[float] = None
        self._interval_indicators: List[Dict[str, float]] = []
        self._indicator_interval_history: deque = deque(maxlen=300)
        self._face_absent_streak_start: Optional[float] = None

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def add_frame_data(self, timestamp: float,
                       face_absent: bool = False,
                       multiple_faces: bool = False,
                       offscreen_gaze: bool = False,
                       suspicious_object: bool = False,
                       reading_pattern: bool = False,
                       looking_away: bool = False,
                       head_movement_stats: Optional[Dict] = None,
                       head_movement_stats_30f: Optional[Dict] = None,
                       temporal_prediction: Optional[object] = None):
        """
        Tambahkan data frame ke interval penilaian saat ini.

        Args:
            timestamp: Timestamp frame
            face_absent: Wajah tidak terdeteksi
            multiple_faces: Lebih dari 1 wajah
            offscreen_gaze: Pandangan di luar layar
            suspicious_object: Objek mencurigakan terdeteksi
            reading_pattern: Pola membaca terdeteksi
            looking_away: Kepala menoleh jauh dari kamera
            head_movement_stats: Statistik pergerakan kepala
            temporal_prediction: Hasil prediksi temporal model
        """
        # Initialize interval
        if self._current_interval_start is None:
            self._current_interval_start = timestamp

        # Build indicator values for this frame
        if face_absent:
            if self._face_absent_streak_start is None:
                self._face_absent_streak_start = timestamp
            absent_duration = timestamp - self._face_absent_streak_start
        else:
            self._face_absent_streak_start = None
            absent_duration = 0.0

        # S3 protocol: treat face absence as suspicious when consecutive absence
        # exceeds configured threshold (default 3 seconds).
        face_absent_score = 1.0 if (
            face_absent and absent_duration >= self.face_absent_consecutive_seconds
        ) else 0.0

        indicators = {
            "face_absent": face_absent_score,
            "multiple_faces": 1.0 if multiple_faces else 0.0,
            "offscreen_gaze": 1.0 if offscreen_gaze else 0.0,
            "suspicious_object": 1.0 if suspicious_object else 0.0,
            "reading_pattern": 1.0 if reading_pattern else 0.0,
            "head_movement_anomaly": 0.0,
            "response_latency": 0.0,
            "temporal_anomaly": 0.0,
        }

        # Head movement anomaly
        if looking_away:
            indicators["head_movement_anomaly"] = 0.5
        if head_movement_stats and head_movement_stats.get("valid"):
            if head_movement_stats.get("is_repetitive", False):
                indicators["head_movement_anomaly"] = 1.0
            elif head_movement_stats.get("yaw_std", 0) > 15:
                indicators["head_movement_anomaly"] = max(
                    indicators["head_movement_anomaly"],
                    min(head_movement_stats["yaw_std"] / 30.0, 1.0)
                )
        if head_movement_stats_30f and head_movement_stats_30f.get("valid"):
            if head_movement_stats_30f.get("yaw_std", 0) > 15:
                indicators["head_movement_anomaly"] = max(
                    indicators["head_movement_anomaly"],
                    min(head_movement_stats_30f["yaw_std"] / 30.0, 1.0)
                )

        # Temporal anomaly from model
        if temporal_prediction is not None:
            temporal_prob = float(
                getattr(temporal_prediction, "cheat_probability",
                        getattr(temporal_prediction, "prob_suspicious", 0.0))
            )
            indicators["temporal_anomaly"] = float(np.clip(temporal_prob, 0.0, 1.0))

        self._interval_indicators.append(indicators)

        # Check if interval is complete
        if timestamp - self._current_interval_start >= self.interval_seconds:
            score = self._compute_interval_score(
                self._current_interval_start, timestamp, temporal_prediction
            )
            self.score_history.append(score)
            self._current_interval_start = timestamp
            self._interval_indicators.clear()
            return score

        return None

    def finalize_interval(self, timestamp: Optional[float] = None,
                          temporal_prediction: Optional[object] = None):
        """
        Finalize interval yang belum tertutup.

        Berguna saat sesi selesai sebelum mencapai `interval_seconds`,
        agar data di interval terakhir tetap menghasilkan skor.
        """
        if not self._interval_indicators:
            return None

        if self._current_interval_start is None:
            interval_start = timestamp if timestamp is not None else 0.0
        else:
            interval_start = self._current_interval_start

        interval_end = timestamp if timestamp is not None else interval_start
        if interval_end < interval_start:
            interval_end = interval_start

        score = self._compute_interval_score(
            interval_start, interval_end, temporal_prediction
        )
        self.score_history.append(score)
        self._current_interval_start = None
        self._interval_indicators.clear()
        return score

    def get_thresholds(self) -> Dict[str, float]:
        """Ambil threshold risk level aktif."""
        if self.risk_scheme == "thesis3":
            return {
                "warning": self.threshold_warning,
                "suspicious": self.threshold_suspicious,
            }
        return {
            "suspicious": self.threshold_suspicious,
            "high_risk": self.threshold_high_risk,
            "critical": self.threshold_critical,
        }

    def _compute_interval_score(self, interval_start: float,
                                 interval_end: float,
                                 temporal_prediction=None) -> CheatingScore:
        """Hitung skor kecurangan untuk satu interval."""
        score = CheatingScore(
            timestamp=interval_end,
            interval_start=interval_start,
            interval_end=interval_end,
        )

        if not self._interval_indicators:
            return score

        # Average indicator values over the interval
        avg_indicators = {}
        for name in self.INDICATOR_NAMES:
            values = [ind[name] for ind in self._interval_indicators]
            avg_indicators[name] = float(np.mean(values))

        score.indicator_scores = avg_indicators
        self._indicator_interval_history.append(avg_indicators.copy())

        # --- Normalize indicator values (min-max) ---
        norm_indicators = self._normalize_indicators_minmax(avg_indicators)
        score.normalized_indicator_scores = norm_indicators

        # --- Effective weights: static * adaptive (frequency-based) ---
        base_weights = self._normalize_weights(self.weights)
        if self.dynamic_weighting:
            adaptive_weights = self._adaptive_weights(avg_indicators)
            combined = {
                name: base_weights.get(name, 0.0) * adaptive_weights.get(name, 0.0)
                for name in self.INDICATOR_NAMES
            }
            eff_weights = self._normalize_weights(combined)
        else:
            eff_weights = base_weights
        score.effective_weights = eff_weights

        # --- Weighted aggregation score (R in [0,1]) ---
        total = 0.0
        for name in self.INDICATOR_NAMES:
            total += eff_weights.get(name, 0.0) * norm_indicators.get(name, 0.0)
        linear_score = float(np.clip(total, 0.0, 1.0))
        if self.use_sigmoid:
            # Sigmoid mapping for probability-style score.
            score.total_score = float(
                1.0 / (1.0 + np.exp(-(self.sigmoid_scale * linear_score + self.sigmoid_bias)))
            )
        else:
            score.total_score = linear_score
        score.score_percent = float(score.total_score * 100.0)

        # Risk level
        if self.risk_scheme == "thesis3":
            if score.total_score >= self.threshold_suspicious:
                score.risk_level = RiskLevel.SUSPICIOUS
            elif score.total_score >= self.threshold_warning:
                score.risk_level = RiskLevel.WARNING
            else:
                score.risk_level = RiskLevel.NORMAL
        else:
            if score.total_score >= self.threshold_critical:
                score.risk_level = RiskLevel.CRITICAL
            elif score.total_score >= self.threshold_high_risk:
                score.risk_level = RiskLevel.HIGH_RISK
            elif score.total_score >= self.threshold_suspicious:
                score.risk_level = RiskLevel.SUSPICIOUS
            else:
                score.risk_level = RiskLevel.NORMAL

        # Top contributing indicators
        sorted_indicators = sorted(
            avg_indicators.items(), key=lambda x: x[1], reverse=True
        )
        score.top_indicators = [
            name for name, val in sorted_indicators if val > 0.1
        ][:3]

        # Temporal prediction
        if temporal_prediction and hasattr(temporal_prediction, 'predicted_label'):
            score.temporal_prediction = temporal_prediction.predicted_label
            score.temporal_confidence = float(
                getattr(temporal_prediction, "cheat_probability",
                        getattr(temporal_prediction, "confidence", 0.0))
            )
            score.prob_third_party = getattr(temporal_prediction, 'prob_third_party', 0.0)
            score.prob_ai_assistance = getattr(temporal_prediction, 'prob_ai_assistance', 0.0)

        return score

    def _normalize_indicators_minmax(self, indicators: Dict[str, float]) -> Dict[str, float]:
        values = np.array([indicators.get(name, 0.0) for name in self.INDICATOR_NAMES], dtype=np.float32)
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmax - vmin < 1e-8:
            # All indicators equal; keep in-range values as-is.
            return {name: float(np.clip(indicators.get(name, 0.0), 0.0, 1.0)) for name in self.INDICATOR_NAMES}
        return {
            name: float((indicators.get(name, 0.0) - vmin) / (vmax - vmin))
            for name in self.INDICATOR_NAMES
        }

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        vec = np.array([max(0.0, float(weights.get(name, 0.0))) for name in self.INDICATOR_NAMES], dtype=np.float32)
        s = float(np.sum(vec))
        if s <= 1e-8:
            uniform = 1.0 / len(self.INDICATOR_NAMES)
            return {name: uniform for name in self.INDICATOR_NAMES}
        return {name: float(vec[i] / s) for i, name in enumerate(self.INDICATOR_NAMES)}

    def _adaptive_weights(self, indicators: Dict[str, float]) -> Dict[str, float]:
        # alpha_i proportional to empirical frequency/intensity over interval history.
        profile = {name: float(indicators.get(name, 0.0)) for name in self.INDICATOR_NAMES}
        if self._indicator_interval_history:
            for name in self.INDICATOR_NAMES:
                history_vals = [it.get(name, 0.0) for it in self._indicator_interval_history]
                profile[name] = float(np.mean(history_vals))
        alpha = {name: profile.get(name, 0.0) + 1e-6 for name in self.INDICATOR_NAMES}
        return self._normalize_weights(alpha)

    def get_overall_score(self) -> Dict:
        """Hitung skor keseluruhan dari semua interval."""
        if not self.score_history:
            return {
                "total_intervals": 0,
                "overall_score": 0.0,
                "overall_score_percent": 0.0,
                "overall_risk": RiskLevel.NORMAL.value,
                "max_score": 0.0,
                "max_score_percent": 0.0,
                "avg_score": 0.0,
                "avg_score_percent": 0.0,
            }

        scores = [s.total_score for s in self.score_history]

        # Count risk levels
        risk_counts = {level.value: 0 for level in RiskLevel}
        for s in self.score_history:
            risk_counts[s.risk_level.value] += 1

        return {
            "total_intervals": len(self.score_history),
            "overall_score": float(np.mean(scores)),
            "overall_score_percent": float(np.mean(scores) * 100.0),
            "overall_risk": self._overall_risk_level(scores).value,
            "max_score": float(np.max(scores)),
            "max_score_percent": float(np.max(scores) * 100.0),
            "avg_score": float(np.mean(scores)),
            "avg_score_percent": float(np.mean(scores) * 100.0),
            "min_score": float(np.min(scores)),
            "std_score": float(np.std(scores)),
            "risk_distribution": risk_counts,
        }

    def _overall_risk_level(self, scores: List[float]) -> RiskLevel:
        """Tentukan risk level keseluruhan."""
        avg = np.mean(scores)
        max_score = np.max(scores)

        if self.risk_scheme == "thesis3":
            if max_score >= self.threshold_suspicious:
                return RiskLevel.SUSPICIOUS
            if avg >= self.threshold_warning:
                return RiskLevel.WARNING
            return RiskLevel.NORMAL

        # Jika pernah critical, overall minimal high_risk
        if max_score >= self.threshold_critical:
            return RiskLevel.HIGH_RISK if avg < self.threshold_high_risk else RiskLevel.CRITICAL
        elif avg >= self.threshold_high_risk:
            return RiskLevel.HIGH_RISK
        elif avg >= self.threshold_suspicious:
            return RiskLevel.SUSPICIOUS
        return RiskLevel.NORMAL

    def reset(self):
        """Reset semua data scoring."""
        self.score_history.clear()
        self._current_interval_start = None
        self._interval_indicators.clear()
        self._indicator_interval_history.clear()
        self._face_absent_streak_start = None
