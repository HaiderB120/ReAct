from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import to_feature_vec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GESTURE_DB = PROJECT_ROOT / "react_gestures.json"


@dataclass(frozen=True)
class GesturePrediction:
    label: str | None
    score: float
    best_label: str | None


def load_gesture_db(path: str | Path = DEFAULT_GESTURE_DB) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_gesture_db(
    db: dict[str, Any],
    path: str | Path = DEFAULT_GESTURE_DB,
) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(db, file, indent=2)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-6 or norm_b < 1e-6:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def build_templates(db: dict[str, Any]) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}

    for label, item in db.items():
        samples = np.asarray(item.get("samples", []), dtype=np.float32)
        if len(samples) == 0:
            continue

        mean_vec = samples.mean(axis=0)
        norm = np.linalg.norm(mean_vec)

        if norm > 1e-6:
            mean_vec = mean_vec / norm

        templates[label] = mean_vec

    return templates


class TemplateGestureRecognizer:
    """
    Simple command recognizer used as the Phase 1 baseline.

    This is intentionally not the future correction-intent model.
    """

    def __init__(
        self,
        threshold: float = 0.92,
        gesture_db_path: str | Path = DEFAULT_GESTURE_DB,
    ) -> None:
        self.threshold = threshold
        self.gesture_db_path = Path(gesture_db_path)
        self.reload()

    def reload(self) -> None:
        self.db = load_gesture_db(self.gesture_db_path)
        self.templates = build_templates(self.db)

    @property
    def labels(self) -> list[str]:
        return sorted(self.templates.keys())

    def predict(self, hand_landmarks, width: int, height: int) -> GesturePrediction:
        if not self.templates:
            return GesturePrediction(label=None, score=0.0, best_label=None)

        feature = to_feature_vec(hand_landmarks, width, height)

        best_label: str | None = None
        best_score = -1.0

        for label, template in self.templates.items():
            score = cosine_similarity(feature, template)
            if score > best_score:
                best_label = label
                best_score = score

        accepted = best_label if best_score >= self.threshold else None

        return GesturePrediction(
            label=accepted,
            score=max(best_score, 0.0),
            best_label=best_label,
        )
