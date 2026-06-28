"""
Modul Pemodelan Temporal
========================
Menganalisis pola perilaku secara sekuensial menggunakan LSTM
untuk membedakan kejadian sesaat dan pola perilaku yang konsisten.

Sesuai dengan BAB 2.6 & 3.4.4 Skripsi:
- Pemodelan Temporal Perilaku Video
- Long Short-Term Memory (LSTM)
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not installed. Temporal model will use rule-based fallback.")


@dataclass
class TemporalPrediction:
    """Hasil prediksi temporal model."""
    frame_number: int = 0
    timestamp: float = 0.0
    # Binary temporal probability (BAB 3.4.4): P in [0, 1]
    cheat_probability: float = 0.0
    predicted_label: str = "normal"  # normal / warning / suspicious
    # Compatibility probabilities used by old reports/UI.
    prob_normal: float = 1.0
    prob_third_party: float = 0.0
    prob_ai_assistance: float = 0.0
    prob_suspicious: float = 0.0
    # Overall flags
    is_cheating_predicted: bool = False
    confidence: float = 0.0


# ============================================================================
# LSTM Model (PyTorch)
# ============================================================================

if TORCH_AVAILABLE:
    class CheatingDetectorLSTM(nn.Module):
        """
        LSTM-based classifier untuk deteksi pola kecurangan temporal.

        Input: sequence of behavior feature vectors
        Output: probability distribution over cheating categories
        """

        def __init__(self, input_size: int = 15, hidden_size: int = 128,
                     num_layers: int = 2, num_outputs: int = 1,
                     dropout: float = 0.3):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # LSTM layers
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )

            # Attention layer
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_outputs),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.

            Args:
                x: (batch, seq_len, input_size)

            Returns:
                (batch, num_outputs) logits
            """
            # LSTM
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)

            # Attention
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_size)

            # Classify
            logits = self.classifier(context)
            return logits

        def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
            """Return cheating probability in [0,1] for inference convenience."""
            logits = self.forward(x)
            return torch.sigmoid(logits)


class TemporalModel:
    """
    Wrapper untuk temporal analysis model.

    Mendukung:
    - LSTM model (jika PyTorch tersedia)
    - Rule-based fallback (jika tidak ada PyTorch atau model belum di-train)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        config = self._load_config(config_path)
        temp_config = config.get("temporal", {})

        self.model_type = temp_config.get("model_type", "lstm")
        self.window_size = temp_config.get("window_size", 30)
        self.hidden_size = temp_config.get("hidden_size", 128)
        self.num_layers = temp_config.get("num_layers", 2)
        self.dropout = temp_config.get("dropout", 0.3)
        self.model_path = temp_config.get("model_path", "models/cheating_detector_lstm.pth")
        self.input_size = 15  # From BehaviorFeatureVector
        self.normal_threshold = float(temp_config.get("normal_threshold", 0.3))
        self.suspicious_threshold = float(temp_config.get("suspicious_threshold", 0.7))

        self.model = None
        self.use_neural = False

        # Try to load neural model
        if TORCH_AVAILABLE:
            self._init_neural_model()

    def _load_config(self, config_path: str) -> dict:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def _init_neural_model(self):
        """Inisialisasi neural model (LSTM)."""
        self.model = CheatingDetectorLSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_outputs=1,
            dropout=self.dropout
        )
        self.model.eval()

        # Check for pre-trained weights
        weight_path = Path(self.model_path)
        if weight_path.exists():
            try:
                payload = torch.load(weight_path, map_location="cpu")
                state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
                self.model.load_state_dict(state_dict)
                self.use_neural = True
                print(f"[INFO] Loaded pre-trained model from {weight_path}")
            except Exception as exc:
                print(f"[WARNING] Failed loading model weights ({exc}). Using rule-based analysis.")
                self.use_neural = False
        else:
            print("[INFO] No pre-trained model found. Using rule-based analysis.")
            self.use_neural = False

    def predict(self, feature_sequence: np.ndarray,
                frame_number: int = 0,
                timestamp: float = 0.0) -> TemporalPrediction:
        """
        Prediksi berdasarkan sequence of behavior features.

        Args:
            feature_sequence: Array (window_size, num_features)
            frame_number: Nomor frame
            timestamp: Timestamp

        Returns:
            TemporalPrediction
        """
        result = TemporalPrediction(
            frame_number=frame_number,
            timestamp=timestamp
        )

        if feature_sequence is None:
            return result

        sequence = np.asarray(feature_sequence, dtype=np.float32)
        if sequence.ndim != 2 or sequence.shape[1] != self.input_size:
            return result
        if sequence.shape[0] < self.window_size:
            return result

        # Use the most recent window for temporal decision.
        sequence = sequence[-self.window_size:]

        if self.use_neural and self.model is not None:
            return self._predict_neural(sequence, result)
        else:
            return self._predict_rule_based(sequence, result)

    def _predict_neural(self, sequence: np.ndarray,
                        result: TemporalPrediction) -> TemporalPrediction:
        """Prediksi menggunakan LSTM model."""
        # Prepare input
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, seq, features)

        with torch.no_grad():
            cheat_prob = self.model.predict_proba(input_tensor).squeeze().item()

        self._fill_prediction_fields(result, float(cheat_prob))

        return result

    def _predict_rule_based(self, sequence: np.ndarray,
                            result: TemporalPrediction) -> TemporalPrediction:
        """
        Prediksi berbasis aturan (fallback ketika belum ada trained model).
        Menganalisis pola dalam sequence untuk mendeteksi kecurangan.
        """
        # Feature indices (from BehaviorFeatureVector; BAB 3.4.4):
        # 0:yaw, 1:pitch, 2:roll, 3:gaze_x, 4:gaze_y,
        # 5:looking_away, 6:offscreen, 7:fixating, 8:saccade,
        # 9:num_faces, 10:suspicious_object, 11:left_ear, 12:right_ear,
        # 13:yaw_velocity, 14:gaze_velocity

        n_frames = len(sequence)

        # Multi-face suspicious ratio.
        multi_face_ratio = float(np.mean(sequence[:, 9] > 1.0))
        looking_away_ratio = float(np.mean(sequence[:, 5]))
        offscreen_ratio = float(np.mean(sequence[:, 6]))
        suspicious_object_ratio = float(np.mean(sequence[:, 10]))
        yaw_std = np.std(sequence[:, 0])
        yaw_vel_mean = float(np.mean(sequence[:, 13]))
        gaze_vel_mean = float(np.mean(sequence[:, 14]))
        fixation_ratio = float(np.mean(sequence[:, 7]))
        saccade_ratio = float(np.mean(sequence[:, 8]))

        # Stabilized rule-based fallback to approximate visual temporal anomaly.
        cheat_score = (
            multi_face_ratio * 0.20
            + looking_away_ratio * 0.18
            + offscreen_ratio * 0.15
            + suspicious_object_ratio * 0.20
            + min(yaw_std / 30.0, 1.0) * 0.10
            + min(yaw_vel_mean / 40.0, 1.0) * 0.07
            + min(gaze_vel_mean / 4.0, 1.0) * 0.05
            + fixation_ratio * 0.03
            + saccade_ratio * 0.02
        )
        cheat_prob = float(np.clip(cheat_score, 0.0, 1.0))
        self._fill_prediction_fields(result, cheat_prob)

        return result

    def _fill_prediction_fields(self, result: TemporalPrediction, cheat_prob: float):
        """Populate TemporalPrediction consistently for neural/rule-based paths."""
        result.cheat_probability = float(np.clip(cheat_prob, 0.0, 1.0))
        result.prob_suspicious = result.cheat_probability
        result.prob_normal = 1.0 - result.cheat_probability
        # Legacy class-wise probabilities are not modeled in binary mode.
        result.prob_third_party = 0.0
        result.prob_ai_assistance = 0.0

        if result.cheat_probability < self.normal_threshold:
            result.predicted_label = "normal"
        elif result.cheat_probability < self.suspicious_threshold:
            result.predicted_label = "warning"
        else:
            result.predicted_label = "suspicious"

        result.confidence = float(max(result.cheat_probability, 1.0 - result.cheat_probability))
        result.is_cheating_predicted = result.predicted_label != "normal"

    def save_model(self, path: str = "models/cheating_detector_lstm.pth"):
        """Save model weights."""
        if self.model is not None and TORCH_AVAILABLE:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), path)
            print(f"[INFO] Model saved to {path}")

    def load_model(self, path: str = "models/cheating_detector_lstm.pth"):
        """Load model weights."""
        if self.model is not None and TORCH_AVAILABLE:
            payload = torch.load(path, map_location="cpu")
            state_dict = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.use_neural = True
            print(f"[INFO] Model loaded from {path}")
