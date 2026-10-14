"""Recognize faces in a single image file."""

import argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Recognize faces in one image")
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="Max confidence to accept a match (default 70)",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save annotated image",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    cascade = load_cascade()
    recognizer = load_recognizer()
    labels = load_labels()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    boxes = detect_faces(gray, cascade)
    if not boxes:
        print("No faces detected.")
        return

    for (x, y, w, h) in boxes:
        face = crop_face(gray, (x, y, w, h))
        name, confidence = predict_name(
            face, recognizer, labels, args.threshold
        )
        print(f"{name} (confidence {confidence:.1f})")
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            image,
            f"{name}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    if args.save:
        cv2.imwrite(str(args.save), image)
        print(f"Saved {args.save}")

    cv2.imshow("Result", image)
    print("Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
