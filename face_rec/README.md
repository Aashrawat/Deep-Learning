# Face Recognition (starter project)

Train a small face recognizer on **your** photos, then recognize you from a webcam or image.

Uses OpenCV **LBPH** (works on Python 3.14; no TensorFlow required).

## Setup

```powershell
cd face_rec
python -m pip install -r requirements.txt
```

## Step 1 — Capture your face photos

```powershell
python capture_photos.py YourName
```

- Look at the camera in good light.
- Press **SPACE** to save each photo (default: 15).
- Press **Q** to quit early.
- Photos go to `dataset/YourName/`.

Add more people by running again with another name.

## Step 2 — Train the model

```powershell
python train.py
```

Creates:

- `models/face_model.yml` — trained recognizer
- `models/labels.json` — person names

## Step 3 — Recognize

**Webcam (live):**

```powershell
python recognize_webcam.py
```

Press **Q** to quit.

**Single image:**

```powershell
python recognize_image.py path\to\photo.jpg
```

If recognition is too loose or too strict, tune:

```powershell
python recognize_webcam.py --threshold 55
```

Lower threshold = stricter (fewer false matches).

## Tips

- Use 10–20 clear front-facing photos per person.
- Retrain after adding photos: `python train.py`
- Only enroll people who gave you permission.

## Upgrade later

For higher accuracy (needs Python 3.11/3.12 + TensorFlow), can switch to **DeepFace** or **InsightFace** later. This project is the simplest path that runs on your current Python.
