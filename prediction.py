from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(__file__).with_name("knn_model.sav")
FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained iris classifier."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict(data):
    """Run prediction on a 2D iterable of feature values."""
    model = load_model()
    frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    return model.predict(frame)
