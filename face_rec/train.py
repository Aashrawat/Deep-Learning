"""Train the face model from dataset/<person_name>/*.jpg photos."""

from pathlib import Path

import cv2
import numpy as np

from config import DATASET_DIR, MODEL_PATH
from face_utils import crop_face, detect_faces, load_cascade, save_labels


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_training_data() -> tuple[list, list, dict[int, str]]:
    cascade = load_cascade()
    faces: list = []
    labels: list = []
    id_to_name: dict[int, str] = {}

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder missing: {DATASET_DIR}\n"
            "Create it with: python capture_photos.py YourName"
        )

    person_dirs = sorted(p for p in DATASET_DIR.iterdir() if p.is_dir())
    if not person_dirs:
        raise FileNotFoundError(
            f"No person folders in {DATASET_DIR}. Run capture_photos.py first."
        )

    for person_id, person_dir in enumerate(person_dirs):
        name = person_dir.name
        id_to_name[person_id] = name
        image_paths = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not image_paths:
            print(f"Warning: no images in {person_dir}")
            continue

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: could not read {image_path}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            boxes = detect_faces(gray, cascade)
            if not boxes:
                print(f"Warning: no face found in {image_path}")
                continue

            # Use the largest detected face
            box = max(boxes, key=lambda b: b[2] * b[3])
            face = crop_face(gray, box)
            faces.append(face)
            labels.append(person_id)
            print(f"  + {name} <- {image_path.name}")

    return faces, labels, id_to_name


def main() -> None:
    print("Collecting faces from dataset/ ...")
    faces, labels, id_to_name = collect_training_data()

    if len(faces) < 2:
        raise RuntimeError(
            "Need at least 2 face samples total. Add more photos per person."
        )

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_ids = np.array(labels, dtype=np.int32)
    recognizer.train(faces, label_ids)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_PATH))
    save_labels(id_to_name)

    print(f"\nTrained on {len(faces)} samples for {len(id_to_name)} people.")
    print(f"Model saved: {MODEL_PATH}")
    print("Next step: python recognize_webcam.py")


if __name__ == "__main__":
    main()
