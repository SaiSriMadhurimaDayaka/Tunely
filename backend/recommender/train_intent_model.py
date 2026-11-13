"""Train/evaluate the small curated intent prototype. Run as a module."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             multilabel_confusion_matrix)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from scipy.sparse import hstack

from .intent_ml import DEFAULT_MODEL_PATH, INTENSITIES, MODEL_VERSION, MOODS, save_bundle
from .intent_training_data import ROOT, write_corpus
from .reviewed_catalog import APPROVED_GENRES


def _read(name: str):
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["labels"] = [genre for genre in row["genres"].split(";") if genre]
    return rows


def _metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    per_genre = {}
    matrix = multilabel_confusion_matrix(truth, pred)
    for index, genre in enumerate(APPROVED_GENRES):
        tn, fp, fn, tp = (int(value) for value in matrix[index].ravel())
        per_genre[genre] = {"precision": float(precision_score(truth[:, index], pred[:, index], zero_division=0)),
                            "recall": float(recall_score(truth[:, index], pred[:, index], zero_division=0)),
                            "f1": float(f1_score(truth[:, index], pred[:, index], zero_division=0)),
                            "support": int(truth[:, index].sum()),
                            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
    return {"subset_accuracy": float(accuracy_score(truth, pred)),
            "precision_micro": float(precision_score(truth, pred, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(truth, pred, average="micro", zero_division=0)),
            "f1_micro": float(f1_score(truth, pred, average="micro", zero_division=0)),
            "precision_macro": float(precision_score(truth, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(truth, pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(truth, pred, average="macro", zero_division=0)),
            "per_genre": per_genre}


def main() -> dict:
    split_counts = write_corpus()
    train, validation, test = (_read(f"{name}.csv") for name in ("train", "validation", "test"))
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                            strip_accents="unicode", max_features=12000)
    mood_encoder = OneHotEncoder(categories=[list(MOODS)], handle_unknown="ignore")
    intensity_encoder = OneHotEncoder(categories=[list(INTENSITIES)], handle_unknown="ignore")
    mlb = MultiLabelBinarizer(classes=list(APPROVED_GENRES))
    mlb.fit([[]])

    # Fit text vocabulary only on training rows; holdout wording is never seen here.
    x_text_train = tfidf.fit_transform([r["preference"] for r in train])
    x_mood_train = mood_encoder.fit_transform(np.asarray([r["mood"] for r in train], dtype=object).reshape(-1, 1))
    x_intensity_train = intensity_encoder.fit_transform(np.asarray([r["intensity"] for r in train], dtype=object).reshape(-1, 1))
    x_train = hstack((x_text_train, x_mood_train, x_intensity_train), format="csr")
    y_train = mlb.transform([r["labels"] for r in train])

    def transform(rows):
        return hstack((tfidf.transform([r["preference"] for r in rows]),
                       mood_encoder.transform(np.asarray([r["mood"] for r in rows], dtype=object).reshape(-1, 1)),
                       intensity_encoder.transform(np.asarray([r["intensity"] for r in rows], dtype=object).reshape(-1, 1))), format="csr")

    x_val, y_val = transform(validation), mlb.transform([r["labels"] for r in validation])
    best = None
    validation_scores = {}
    for c in (0.25, 0.5, 1.0, 2.0, 4.0):
        classifier = OneVsRestClassifier(LogisticRegression(C=c, max_iter=1500, class_weight="balanced", random_state=42))
        classifier.fit(x_train, y_train)
        val_pred = classifier.predict(x_val)
        score = f1_score(y_val, val_pred, average="macro", zero_division=0)
        validation_scores[str(c)] = float(score)
        if best is None or score > best[0]:
            best = score, c

    # Final evaluation on the independent test file, still not used for fitting or selection.
    classifier = OneVsRestClassifier(LogisticRegression(C=best[1], max_iter=1500,
                                                       class_weight="balanced", random_state=42))
    classifier.fit(x_train, y_train)
    x_test, y_test = transform(test), mlb.transform([r["labels"] for r in test])
    train_pred = classifier.predict(x_train)
    test_pred = classifier.predict(x_test)

    forest = OneVsRestClassifier(RandomForestClassifier(n_estimators=180, min_samples_leaf=2,
                                                        class_weight="balanced", random_state=42, n_jobs=1))
    forest.fit(x_train, y_train)
    forest_test = forest.predict(x_test)

    # Transparent existing-style mood-only baseline (preference is ignored).
    mood_labels = {"happy": {"Party"}, "sad": {"Sad"}, "calm": {"Devotional", "Romantic", "Sad"},
                   "energetic": {"Party", "Mass"}, "stressed": {"Devotional", "Sad", "Romantic"},
                   "romantic": {"Romantic"}, "neutral": set(APPROVED_GENRES),
                   "motivated": {"Mass", "Party"}, "nostalgic": {"Old"}}
    rule_pred = np.asarray([[int(label in mood_labels[row["mood"]]) for label in APPROVED_GENRES] for row in test])

    bundle = {"model_version": MODEL_VERSION, "training_dataset_version": "curated-intents-2026-11-v1",
              "trained_at": datetime.now(timezone.utc).isoformat(), "genre_order": tuple(APPROVED_GENRES),
              "tfidf": tfidf, "mood_encoder": mood_encoder, "intensity_encoder": intensity_encoder,
              "label_binarizer": mlb, "classifier": classifier, "validation_best_c": best[1],
              "training_examples": len(train), "validation_examples": len(validation), "test_examples": len(test),
              "corpus_note": "Synthetic/curated editorial intent examples; not observed user behavior."}
    save_bundle(bundle)

    report = {"model": "TF-IDF + sparse one-hot mood/intensity + OneVsRest LogisticRegression",
              "selected_c": best[1], "validation_macro_f1_by_c": validation_scores,
              "training_example_count": len(train), "validation_example_count": len(validation),
              "test_example_count": len(test), "class_coverage": {g: int(y_train[:, i].sum()) for i, g in enumerate(APPROVED_GENRES)},
              "training_metrics": _metrics(y_train, train_pred), "test_metrics": _metrics(y_test, test_pred),
              "baseline_comparison_test": {"rule_based_mood_only": _metrics(y_test, rule_pred),
                                            "random_forest_ovr": _metrics(y_test, forest_test)},
              "limitations": ["All examples are synthetic/curated and do not represent actual listener behavior.",
                              "This small controlled holdout does not establish real-world accuracy.",
                              "Only the six manually reviewed genre labels are predicted; song-level labels remain unchanged."],
              "corpus_counts": split_counts, "model_path": str(DEFAULT_MODEL_PATH)}
    report_path = ROOT / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
