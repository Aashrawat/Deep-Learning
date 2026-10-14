"""Capture enrollment photos from your webcam into dataset/<your_name>/."""

import argparse
import time
from pathlib import Path

import cv2

from config import DATASET_DIR
from face_utils import detect_faces, load_cascade


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture face photos for training")
    parser.add_argument("name", help="Person name (folder name), e.g. Alice")
    parser.add_argument(
        "--count",
        type=int,
        default=15,
        help="Number of photos to capture (default: 15)",
    )
    args = parser.parse_args()

    person_dir = DATASET_DIR / args.name.replace(" ", "_")
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions.")

    cascade = load_cascade()
    print(f"Saving to: {person_dir}")
    print("Face the camera. Wait for the GREEN box, then press SPACE. Q to quit.")

    captured = 0
    while captured < args.count:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        preview = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = detect_faces(gray, cascade)
        face_ok = len(boxes) > 0
        for (x, y, w, h) in boxes:
            color = (0, 255, 0) if face_ok else (0, 0, 255)
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)

        status = "Face OK - press SPACE" if face_ok else "No face - move closer / better light"
        cv2.putText(
            preview,
            f"{captured}/{args.count}  {status}  Q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if face_ok else (0, 0, 255),
            2,
        )
        cv2.imshow("Capture enrollment photos", preview)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key == ord(" "):
            if not face_ok:
                print("Skipped: no face detected. Adjust position and try again.")
                continue
            path = person_dir / f"photo_{captured + 1:02d}.jpg"
            cv2.imwrite(str(path), frame)
            captured += 1
            print(f"Saved {path.name}")
            time.sleep(0.3)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Captured {captured} photos.")
    if captured > 0:
        print("Next step: python train.py")


if __name__ == "__main__":
    main()
