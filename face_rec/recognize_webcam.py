"""Live face recognition from webcam using your trained model."""

import argparse

import cv2

from face_utils import (
    crop_face,
    detect_faces,
    load_cascade,
    load_labels,
    load_recognizer,
    predict_name,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize faces from webcam")
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Max confidence to accept a match (lower=stricter, default 70)",
    )
    args = parser.parse_args()

    cascade = load_cascade()
    recognizer = load_recognizer()
    labels = load_labels()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("To quit: click the VIDEO window, then press Q or ESC.")
    print("Or press Ctrl+C in this terminal.")

    window = "Face recognition"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x, y, w, h) in detect_faces(gray, cascade):
                face = crop_face(gray, (x, y, w, h))
                name, confidence = predict_name(
                    face, recognizer, labels, args.threshold
                )
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{name} ({confidence:.0f})"
                cv2.putText(
                    frame,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                "Quit: Q or ESC (click this window first)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow(window, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # q, Q, or ESC
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
