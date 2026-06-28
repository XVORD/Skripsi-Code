"""
Modul Deteksi Wajah dan Landmark
================================
Melakukan deteksi wajah, ekstraksi 468 facial landmarks menggunakan MediaPipe Face Mesh,
serta deteksi multi-face dan face absence.

Sesuai dengan BAB 3.4.2 Skripsi: Modul Deteksi Wajah dan Objek
"""

import cv2
import numpy as np
import mediapipe as mp
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class FaceDetectionResult:
    """Hasil deteksi wajah untuk satu frame."""
    frame_number: int = 0
    timestamp: float = 0.0
    num_faces: int = 0
    face_present: bool = False
    faces: List[Dict[str, Any]] = field(default_factory=list)
    # Flags
    multiple_faces: bool = False
    face_absent: bool = False


class FaceDetector:
    """
    Detektor wajah dan facial landmarks menggunakan MediaPipe Face Mesh.

    Kemampuan:
    - Deteksi wajah dengan 468 landmark points
    - Multi-face detection (deteksi orang lain di frame)
    - Face absence detection (kandidat hilang dari kamera)
    - Bounding box extraction
    - Iris landmark untuk eye gaze estimation
    """

    # Indeks landmark penting
    # Nose tip, chin, left eye corner, right eye corner, left mouth, right mouth
    KEY_LANDMARK_IDS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "left_eye_inner": 133,
        "right_eye_outer": 362,
        "right_eye_inner": 263,
        "left_mouth": 61,
        "right_mouth": 291,
        "forehead": 10,
        "left_cheek": 234,
        "right_cheek": 454,
    }

    # Landmark points untuk Head Pose Estimation (PnP solver).
    # First 6 keep the historical order; the extra points let the head-pose
    # estimator use a more stable extended model when enabled.
    POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291, 133, 362, 10, 234, 454]

    # Iris landmarks (MediaPipe Face Mesh with refine_landmarks=True)
    LEFT_IRIS_IDS = [468, 469, 470, 471, 472]
    RIGHT_IRIS_IDS = [473, 474, 475, 476, 477]

    # Eye contour landmarks
    LEFT_EYE_IDS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_IDS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Inisialisasi FaceDetector.

        Args:
            config_path: Path ke file konfigurasi YAML
        """
        config = self._load_config(config_path)
        face_config = config.get("face_detection", {})

        self.min_detection_confidence = face_config.get("min_detection_confidence", 0.7)
        self.min_tracking_confidence = face_config.get("min_tracking_confidence", 0.5)
        self.max_num_faces = face_config.get("max_num_faces", 3)

        # Initialize MediaPipe Face Mesh
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
            raise RuntimeError(
                "MediaPipe Face Mesh API is unavailable. "
                "Install a compatible `mediapipe` package that provides `solutions.face_mesh`."
            )
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,  # Enable iris landmarks
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def _load_config(self, config_path: str) -> dict:
        """Load konfigurasi dari file YAML."""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def detect(self, frame_rgb: np.ndarray, frame_number: int = 0,
               timestamp: float = 0.0) -> FaceDetectionResult:
        """
        Deteksi wajah dan landmarks pada satu frame.

        Args:
            frame_rgb: Frame dalam format RGB (dari MediaPipe requirement)
            frame_number: Nomor frame
            timestamp: Timestamp dalam detik

        Returns:
            FaceDetectionResult dengan informasi semua wajah terdeteksi
        """
        result = FaceDetectionResult(
            frame_number=frame_number,
            timestamp=timestamp
        )

        h, w, _ = frame_rgb.shape

        # Process dengan MediaPipe
        mp_results = self.face_mesh.process(frame_rgb)

        if mp_results.multi_face_landmarks is None:
            result.face_present = False
            result.face_absent = True
            result.num_faces = 0
            return result

        num_faces = len(mp_results.multi_face_landmarks)
        result.num_faces = num_faces
        result.face_present = num_faces > 0
        result.face_absent = num_faces == 0
        result.multiple_faces = num_faces > 1

        for face_idx, face_landmarks in enumerate(mp_results.multi_face_landmarks):
            face_data = self._extract_face_data(face_landmarks, face_idx, w, h)
            result.faces.append(face_data)

        return result

    def _extract_face_data(self, face_landmarks, face_idx: int,
                           img_w: int, img_h: int) -> Dict[str, Any]:
        """
        Ekstrak data detail dari satu wajah.

        Returns:
            Dictionary berisi landmarks, bounding box, key points, dll.
        """
        # Semua landmarks sebagai array numpy (normalized)
        all_landmarks = []
        for lm in face_landmarks.landmark:
            all_landmarks.append([lm.x, lm.y, lm.z])
        landmarks_array = np.array(all_landmarks)

        # Landmarks dalam pixel coordinates
        landmarks_pixel = landmarks_array.copy()
        landmarks_pixel[:, 0] *= img_w
        landmarks_pixel[:, 1] *= img_h

        # Bounding box
        x_coords = landmarks_pixel[:, 0]
        y_coords = landmarks_pixel[:, 1]
        bbox = {
            "x_min": int(np.min(x_coords)),
            "y_min": int(np.min(y_coords)),
            "x_max": int(np.max(x_coords)),
            "y_max": int(np.max(y_coords)),
        }

        # Key landmarks untuk Head Pose (pixel coordinates, float)
        pose_landmarks_2d = []
        pose_landmarks_3d = []
        for lid in self.POSE_LANDMARK_IDS:
            lm = face_landmarks.landmark[lid]
            pose_landmarks_2d.append([lm.x * img_w, lm.y * img_h])
            pose_landmarks_3d.append([lm.x * img_w, lm.y * img_h, lm.z * 3000])

        # Iris center points (jika tersedia, refine_landmarks=True)
        left_iris_center = None
        right_iris_center = None
        try:
            left_iris_pts = []
            for lid in self.LEFT_IRIS_IDS:
                lm = face_landmarks.landmark[lid]
                left_iris_pts.append([lm.x * img_w, lm.y * img_h])
            left_iris_center = np.mean(left_iris_pts, axis=0).tolist()

            right_iris_pts = []
            for lid in self.RIGHT_IRIS_IDS:
                lm = face_landmarks.landmark[lid]
                right_iris_pts.append([lm.x * img_w, lm.y * img_h])
            right_iris_center = np.mean(right_iris_pts, axis=0).tolist()
        except (IndexError, ValueError):
            pass

        # Eye contour points
        left_eye_pts = []
        for lid in self.LEFT_EYE_IDS:
            lm = face_landmarks.landmark[lid]
            left_eye_pts.append([lm.x * img_w, lm.y * img_h])

        right_eye_pts = []
        for lid in self.RIGHT_EYE_IDS:
            lm = face_landmarks.landmark[lid]
            right_eye_pts.append([lm.x * img_w, lm.y * img_h])

        return {
            "face_idx": face_idx,
            "img_w": img_w,
            "img_h": img_h,
            "landmarks_normalized": landmarks_array,
            "landmarks_pixel": landmarks_pixel,
            "bbox": bbox,
            "pose_landmarks_2d": np.array(pose_landmarks_2d, dtype=np.float64),
            "pose_landmarks_3d": np.array(pose_landmarks_3d, dtype=np.float64),
            "left_iris_center": left_iris_center,
            "right_iris_center": right_iris_center,
            "left_eye_contour": np.array(left_eye_pts),
            "right_eye_contour": np.array(right_eye_pts),
            "raw_landmarks": face_landmarks,
        }

    def draw_detection(self, frame_bgr: np.ndarray, result: FaceDetectionResult,
                       draw_mesh: bool = False, draw_bbox: bool = True,
                       draw_iris: bool = True) -> np.ndarray:
        """
        Gambar hasil deteksi pada frame untuk visualisasi.

        Args:
            frame_bgr: Frame BGR untuk digambar
            result: Hasil deteksi
            draw_mesh: Gambar full face mesh
            draw_bbox: Gambar bounding box
            draw_iris: Gambar iris center

        Returns:
            Frame dengan visualisasi
        """
        output = frame_bgr.copy()

        # Status text
        status_color = (0, 255, 0) if result.face_present else (0, 0, 255)
        status_text = f"Faces: {result.num_faces}"
        if result.face_absent:
            status_text += " [ABSENT!]"
        if result.multiple_faces:
            status_text += " [MULTIPLE!]"

        cv2.putText(output, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        for face_data in result.faces:
            # Bounding box
            if draw_bbox:
                bb = face_data["bbox"]
                color = (0, 255, 0) if face_data["face_idx"] == 0 else (0, 165, 255)
                cv2.rectangle(output,
                              (bb["x_min"], bb["y_min"]),
                              (bb["x_max"], bb["y_max"]),
                              color, 2)

            # Face mesh
            if draw_mesh and "raw_landmarks" in face_data:
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_data["raw_landmarks"],
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # Iris
            if draw_iris:
                if face_data["left_iris_center"]:
                    pt = tuple(int(v) for v in face_data["left_iris_center"])
                    cv2.circle(output, pt, 3, (255, 0, 0), -1)
                if face_data["right_iris_center"]:
                    pt = tuple(int(v) for v in face_data["right_iris_center"])
                    cv2.circle(output, pt, 3, (255, 0, 0), -1)

        return output

    def release(self):
        """Release MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()


# --- Quick Test ---
if __name__ == "__main__":
    from video_acquisition import VideoAcquisition

    print("=== Test Face Detection ===")

    va = VideoAcquisition()
    fd = FaceDetector()

    if va.start("webcam"):
        for frame_num, original, preprocessed in va.stream_frames(max_frames=300):
            # Deteksi wajah
            result = fd.detect(preprocessed, frame_num, frame_num / 15.0)

            # Visualisasi
            display = fd.draw_detection(original, result, draw_mesh=False, draw_bbox=True)
            cv2.imshow("Face Detection Test", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        va.stop()
        fd.release()
        cv2.destroyAllWindows()
