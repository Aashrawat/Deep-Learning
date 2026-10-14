import json
from pathlib import Path

import cv2
import numpy as np

from config import FACE_SIZE, LABELS_PATH, MODEL_PATH, MODELS_DIR


def cascade_path() -> str:
    return str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")


def load_cascade() -> cv2.CascadeClassifier:
    path = cascade_path()
    cascade = cv2.CascadeClassifier(path)
    if cascade.empty():
        raise RuntimeError(f"Could not load face detector: {path}")
    return cascade


def prepare_gray(gray: np.ndarray) -> np.ndarray:
    """Improve contrast so the Haar detector finds faces more reliably."""
    return cv2.equalizeHist(gray)


def detect_faces(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    enhanced = prepare_gray(gray)
    faces = cascade.detectMultiScale(
        enhanced,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        return []
    return [tuple(int(v) for v in f) for f in faces]


def crop_face(gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    face = gray[y : y + h, x : x + w]
    return cv2.resize(face, FACE_SIZE)


def load_labels() -> dict[int, str]:
    if not LABELS_PATH.exists():
        return {}
    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data["id_to_name"].items()}


def save_labels(id_to_name: dict[int, str]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"id_to_name": {str(k): v for k, v in id_to_name.items()}}
    LABELS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_recognizer() -> cv2.face_LBPHFaceRecognizer:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}. Run: python train.py"
        )
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))
    return recognizer


def predict_name(
    face_gray: np.ndarray,
    recognizer: cv2.face_LBPHFaceRecognizer,
    labels: dict[int, str],
    confidence_threshold: float = 70.0,
) -> tuple[str, float]:
    """Lower confidence score = better match (OpenCV LBPH convention)."""
    label_id, confidence = recognizer.predict(face_gray)
    if confidence > confidence_threshold:
        return "Unknown", float(confidence)
    return labels.get(label_id, "Unknown"), float(confidence)
