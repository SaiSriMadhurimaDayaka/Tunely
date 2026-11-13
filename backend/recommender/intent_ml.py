"""Small, optional multi-label model for user listening intent.

The reviewed song catalog supplies song genre labels, not user-intent labels.
Training examples therefore live in a separate, explicitly synthetic curated
corpus. Evaluation is a controlled smoke/evaluation set, not evidence of
real-world recommendation accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reviewed_catalog import APPROVED_GENRES

MODEL_VERSION = "intent-tfidf-ovr-1.0"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "models" / "intent_genre_model.joblib"
MOODS = ("happy", "sad", "calm", "energetic", "stressed", "romantic", "neutral", "motivated", "nostalgic")
INTENSITIES = ("mild", "moderate", "intense")


def _features(bundle: dict[str, Any], preferences: list[str], moods: list[str], intensities: list[str]):
    import numpy as np
    from scipy.sparse import hstack
    text = bundle["tfidf"].transform([value or "" for value in preferences])
    mood = bundle["mood_encoder"].transform(np.asarray(moods, dtype=object).reshape(-1, 1))
    intensity = bundle["intensity_encoder"].transform(np.asarray(intensities, dtype=object).reshape(-1, 1))
    return hstack((text, mood, intensity), format="csr")


class IntentGenreModel:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "IntentGenreModel":
        import joblib
        bundle = joblib.load(path)
        if bundle.get("model_version") != MODEL_VERSION or tuple(bundle.get("genre_order", ())) != APPROVED_GENRES:
            raise ValueError("Intent model version or genre order is incompatible.")
        return cls(bundle)

    def predict_scores(self, preference: str, mood: str, intensity: str) -> dict[str, float]:
        features = _features(self.bundle, [preference], [mood.casefold()], [intensity.casefold()])
        scores = self.bundle["classifier"].predict_proba(features)[0]
        return {genre: float(score) for genre, score in zip(self.bundle["genre_order"], scores)}


def load_intent_model(path: str | Path = DEFAULT_MODEL_PATH) -> IntentGenreModel | None:
    """Load once, returning None when optional ML dependencies/artifact fail."""
    try:
        return IntentGenreModel.load(path)
    except Exception:
        return None


def save_bundle(bundle: dict[str, Any], path: str | Path = DEFAULT_MODEL_PATH) -> None:
    import joblib
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
