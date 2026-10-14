from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "face_model.yml"
LABELS_PATH = MODELS_DIR / "labels.json"
FACE_SIZE = (200, 200)
