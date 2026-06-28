"""
Modul Deteksi Objek
===================
Mendeteksi objek asing/mencurigakan dalam frame video menggunakan YOLOv8,
seperti ponsel, earphone, buku, dan perangkat elektronik lainnya.

Sesuai dengan BAB 3.4.2 Skripsi: Modul Deteksi Wajah dan Objek
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class DetectedObject:
    """Representasi satu objek yang terdeteksi."""
    class_name: str = ""
    confidence: float = 0.0
    bbox: Dict[str, int] = field(default_factory=dict)  # x1, y1, x2, y2
    center: tuple = (0, 0)


@dataclass
class ObjectDetectionResult:
    """Hasil deteksi objek untuk satu frame."""
    frame_number: int = 0
    timestamp: float = 0.0
    objects: List[DetectedObject] = field(default_factory=list)
    suspicious_objects: List[DetectedObject] = field(default_factory=list)
    has_suspicious_object: bool = False


class ObjectDetector:
    """
    Detektor objek mencurigakan menggunakan YOLOv8.

    Target objek yang dideteksi:
    - cell phone (ponsel)
    - book (buku)
    - laptop (laptop tambahan)
    - earbuds/earphone (jika ada custom model)
    """

    # COCO class mapping untuk objek yang relevan
    SUSPICIOUS_CLASSES = {
        "cell phone": "Ponsel",
        "smartphone": "Ponsel",
        "book": "Buku/Catatan",
        "notebook": "Buku/Catatan",
        "earphone": "Earphone/Headset",
        "earphones": "Earphone/Headset",
        "headphone": "Earphone/Headset",
        "headphones": "Earphone/Headset",
        "earbud": "Earphone/Headset",
        "earbuds": "Earphone/Headset",
        "laptop": "Laptop Tambahan",
        "remote": "Remote/Device",
        "tablet": "Tablet",
    }

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Inisialisasi ObjectDetector.

        Args:
            config_path: Path ke file konfigurasi YAML
        """
        config = self._load_config(config_path)
        obj_config = config.get("object_detection", {})

        self.model_path = obj_config.get("model", "yolov8n.pt")
        self.confidence_threshold = obj_config.get("confidence_threshold", 0.5)
        self.target_classes = obj_config.get("target_classes", list(self.SUSPICIOUS_CLASSES.keys()))

        self.model = None
        self._initialized = False

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        return str(name or "").strip().lower().replace("-", " ").replace("_", " ")

    def _is_suspicious_class(self, class_name: str) -> bool:
        cname = self._normalize_class_name(class_name)
        targets = {self._normalize_class_name(t) for t in self.target_classes}
        if cname in targets:
            return True
        # Keyword-level fallback for custom label variants.
        keyword_hits = (
            "phone" in cname
            or "book" in cname
            or "notebook" in cname
            or "earphone" in cname
            or "headphone" in cname
            or "earbud" in cname
        )
        return keyword_hits

    def _load_config(self, config_path: str) -> dict:
        """Load konfigurasi dari file YAML."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def initialize(self) -> bool:
        """
        Inisialisasi model YOLO. Dipanggil terpisah agar loading bisa di-control.

        Returns:
            True jika berhasil
        """
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            self._initialized = True
            print(f"[INFO] YOLOv8 model loaded: {self.model_path}")
            return True
        except ImportError:
            print("[WARNING] ultralytics not installed. Object detection disabled.")
            print("[WARNING] Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"[ERROR] Failed to load YOLO model: {e}")
            return False

    def detect(self, frame_bgr: np.ndarray, frame_number: int = 0,
               timestamp: float = 0.0) -> ObjectDetectionResult:
        """
        Deteksi objek pada satu frame.

        Args:
            frame_bgr: Frame dalam format BGR
            frame_number: Nomor frame
            timestamp: Timestamp dalam detik

        Returns:
            ObjectDetectionResult
        """
        result = ObjectDetectionResult(
            frame_number=frame_number,
            timestamp=timestamp
        )

        if not self._initialized or self.model is None:
            return result

        # Run YOLO inference
        detections = self.model(frame_bgr, verbose=False, conf=self.confidence_threshold)

        for det in detections:
            if det.boxes is None:
                continue

            for box in det.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                obj = DetectedObject(
                    class_name=class_name,
                    confidence=confidence,
                    bbox={"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    center=center
                )

                result.objects.append(obj)

                # Check if suspicious
                if self._is_suspicious_class(class_name):
                    result.suspicious_objects.append(obj)
                    result.has_suspicious_object = True

        return result

    def draw_detection(self, frame_bgr: np.ndarray,
                       result: ObjectDetectionResult) -> np.ndarray:
        """
        Gambar hasil deteksi objek pada frame.

        Args:
            frame_bgr: Frame BGR
            result: Hasil deteksi

        Returns:
            Frame dengan visualisasi
        """
        output = frame_bgr.copy()

        for obj in result.objects:
            bb = obj.bbox
            is_suspicious = obj in result.suspicious_objects
            color = (0, 0, 255) if is_suspicious else (200, 200, 200)
            thickness = 2 if is_suspicious else 1

            # Bounding box
            cv2.rectangle(output, (bb["x1"], bb["y1"]), (bb["x2"], bb["y2"]),
                          color, thickness)

            # Label
            label = f"{obj.class_name}: {obj.confidence:.2f}"
            if is_suspicious:
                label = f"[!] {label}"

            label_y = max(bb["y1"] - 10, 20)
            cv2.putText(output, label, (bb["x1"], label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Status
        if result.has_suspicious_object:
            cv2.putText(output, "SUSPICIOUS OBJECT DETECTED!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return output


# --- Quick Test ---
if __name__ == "__main__":
    from video_acquisition import VideoAcquisition

    print("=== Test Object Detection ===")

    va = VideoAcquisition()
    od = ObjectDetector()

    if not od.initialize():
        print("Object detection not available.")
        exit()

    if va.start("webcam"):
        for frame_num, original, preprocessed in va.stream_frames(max_frames=300):
            # Deteksi objek
            result = od.detect(original, frame_num, frame_num / 15.0)

            # Visualisasi
            display = od.draw_detection(original, result)
            cv2.imshow("Object Detection Test", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        va.stop()
        cv2.destroyAllWindows()
