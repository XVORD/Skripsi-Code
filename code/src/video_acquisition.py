"""
Modul Akuisisi Video
====================
Menangkap video real-time dari webcam atau membaca file video,
lalu mengekstrak frame-frame individual untuk diproses oleh modul selanjutnya.

Sesuai dengan BAB 3.4.1 Skripsi: Modul Akuisisi Video
"""

import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from typing import Optional, Generator, Tuple


class VideoAcquisition:
    """
    Kelas utama untuk akuisisi video dari webcam atau file video.

    Mendukung:
    - Real-time webcam capture
    - Video file playback
    - Configurable FPS dan resolusi
    - Frame preprocessing (resize, normalization)
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Inisialisasi VideoAcquisition dengan konfigurasi.

        Args:
            config_path: Path ke file konfigurasi YAML
        """
        self.config = self._load_config(config_path)
        video_config = self.config.get("video", {})

        self.source = video_config.get("source", "webcam")
        self.camera_id = video_config.get("camera_id", 0)
        self.target_fps = video_config.get("fps", 15)
        self.width = video_config.get("width", 640)
        self.height = video_config.get("height", 480)

        self.cap: Optional[cv2.VideoCapture] = None
        self.actual_fps: float = 0.0
        self.frame_count: int = 0
        self.is_running: bool = False

    def _load_config(self, config_path: str) -> dict:
        """Load konfigurasi dari file YAML."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def start(self, source: Optional[str] = None) -> bool:
        """
        Mulai capture video.

        Args:
            source: Override sumber video ('webcam' atau path file video).
                    Jika None, gunakan konfigurasi default.

        Returns:
            True jika berhasil membuka sumber video
        """
        if source:
            self.source = source

        if self.source == "webcam":
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        else:
            # Buka file video
            video_path = Path(self.source)
            if not video_path.exists():
                print(f"[ERROR] Video file not found: {self.source}")
                return False
            self.cap = cv2.VideoCapture(str(video_path))

        if not self.cap.isOpened():
            print(f"[ERROR] Gagal membuka sumber video: {self.source}")
            return False

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        self.frame_count = 0
        self.is_running = True

        # Info
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Video source opened: {self.source}")
        print(f"[INFO] Resolution: {w}x{h}, FPS: {self.actual_fps:.1f}")

        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Baca satu frame dari sumber video.

        Returns:
            Tuple (success: bool, frame: np.ndarray atau None)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return True, frame
        else:
            self.is_running = False
            return False, None

    def read_frame_preprocessed(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Baca frame dan lakukan preprocessing.

        Returns:
            Tuple (success, original_frame, preprocessed_frame)
            - original_frame: frame asli BGR
            - preprocessed_frame: frame yang sudah di-resize dan dikonversi ke RGB
        """
        ret, frame = self.read_frame()
        if not ret or frame is None:
            return False, None, None

        # Resize jika perlu
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            preprocessed = cv2.resize(frame, (self.width, self.height))
        else:
            preprocessed = frame.copy()

        # Convert BGR ke RGB (dibutuhkan oleh MediaPipe)
        preprocessed_rgb = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        return True, frame, preprocessed_rgb

    def stream_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray, np.ndarray], None, None]:
        """
        Generator yang menghasilkan frame secara terus-menerus.

        Args:
            max_frames: Batas jumlah frame (None = unlimited)

        Yields:
            Tuple (frame_number, original_frame_bgr, preprocessed_frame_rgb)
        """
        frame_interval = 1.0 / self.target_fps if self.source == "webcam" else 0

        while self.is_running:
            start_time = time.time()

            ret, original, preprocessed = self.read_frame_preprocessed()
            if not ret:
                break

            yield self.frame_count, original, preprocessed

            if max_frames and self.frame_count >= max_frames:
                break

            # FPS control untuk webcam
            if frame_interval > 0:
                elapsed = time.time() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def get_video_info(self) -> dict:
        """Dapatkan informasi tentang sumber video yang sedang dibuka."""
        if self.cap is None:
            return {}
        return {
            "source": self.source,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.actual_fps,
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source != "webcam" else -1,
            "frames_read": self.frame_count,
        }

    def stop(self):
        """Hentikan capture dan release resources."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print(f"[INFO] Video capture stopped. Total frames: {self.frame_count}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# --- Quick Test ---
if __name__ == "__main__":
    print("=== Test Video Acquisition ===")

    # Test dengan webcam
    va = VideoAcquisition()
    if va.start("webcam"):
        print(f"Video Info: {va.get_video_info()}")

        for frame_num, original, preprocessed in va.stream_frames(max_frames=100):
            # Tampilkan frame
            cv2.putText(original, f"Frame: {frame_num}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Acquisition Test", original)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        va.stop()
        cv2.destroyAllWindows()
