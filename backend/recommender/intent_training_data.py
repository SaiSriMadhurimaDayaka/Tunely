"""Build the versioned intent corpus from transparent editorial annotations.

This is a small synthetic/curated prototype corpus, not observed user behavior.
Each source example is deliberately associated with an explicit intent and
reviewed multi-label target. Test wording is held out from fitting.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .reviewed_catalog import APPROVED_GENRES
from .intent_ml import MOODS, INTENSITIES

ROOT = Path(__file__).resolve().parents[2] / "data" / "intent"
MOOD_LABELS = {
    "happy": ("Party",), "sad": ("Sad",), "calm": ("Devotional",),
    "energetic": ("Party", "Mass"), "stressed": ("Devotional",),
    "romantic": ("Romantic",), "neutral": (), "motivated": ("Mass", "Party"),
    "nostalgic": ("Old",),
}
TRAIN_INTENTS = (
    ("heartbreak", "heartbreak and separation songs", "songs about a broken heart", ("Sad",)),
    ("dance", "dance and celebration songs", "rhythmic party music", ("Party", "Mass")),
    ("peaceful", "peaceful gentle melodies", "quiet music to unwind", ("Devotional",)),
    ("devotional", "devotional prayer songs", "spiritual worship music", ("Devotional",)),
    ("powerful", "powerful hero elevation songs", "bold mass action songs", ("Mass", "Party")),
    ("romance", "romantic love melodies", "tender songs about affection", ("Romantic",)),
    ("classic", "old nostalgic melodies", "classic songs from earlier years", ("Old",)),
    ("sad", "emotional songs about loneliness", "grief and sorrow melodies", ("Sad",)),
)
TEST_INTENTS = (
    ("heartbreak", "music for separation and emotional loss", ("Sad",)),
    ("dance", "upbeat tracks for dancing at a celebration", ("Party", "Mass")),
    ("peaceful", "soft calming music for a quiet evening", ("Devotional",)),
    ("devotional", "songs of faith and prayer", ("Devotional",)),
    ("powerful", "grand high impact hero entry music", ("Mass", "Party")),
    ("romance", "warm melodies expressing love and devotion", ("Romantic",)),
    ("classic", "memories of vintage songs from the past", ("Old",)),
    ("sad", "lonely grief filled music", ("Sad",)),
)


def examples(split: str) -> list[dict[str, str]]:
    rows = []
    if split == "train":
        for mood in MOODS:
            for intensity in INTENSITIES:
                for intent_key, first, second, labels in TRAIN_INTENTS:
                    for phrase in (first, second):
                        target = set(MOOD_LABELS[mood]) | set(labels)
                        rows.append({"preference": phrase, "mood": mood, "intensity": intensity,
                                     "genres": ";".join(g for g in APPROVED_GENRES if g in target),
                                     "split": split, "intent_group": intent_key})
        return rows

    # Separate compact holdouts cover every mood/intensity pair. They use
    # distinct wording templates from train and from each other.
    intent_sets = TEST_INTENTS if split == "test" else tuple(
        (key, phrase.replace("music", "songs").replace("songs", "melodies").replace("tracks", "tunes"), labels)
        for key, phrase, labels in TEST_INTENTS
    )
    for mood_index, mood in enumerate(MOODS):
        for intensity_index, intensity in enumerate(INTENSITIES):
            index = (mood_index * 3 + intensity_index + (2 if split == "test" else 0)) % len(intent_sets)
            intent_key, phrase, labels = intent_sets[index]
            if split == "test":
                preference = f"Looking for {phrase} with {intensity} intensity"
            else:
                preference = f"Please find {phrase} while I feel {mood}, intensity {intensity}"
            target = set(MOOD_LABELS[mood]) | set(labels)
            rows.append({"preference": preference, "mood": mood, "intensity": intensity,
                         "genres": ";".join(g for g in APPROVED_GENRES if g in target),
                         "split": split, "intent_group": intent_key})
    return rows


def write_corpus() -> dict[str, int]:
    ROOT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, filename in (("train", "train.csv"), ("validation", "validation.csv"), ("test", "test.csv")):
        rows = examples(split)
        path = ROOT / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("preference", "mood", "intensity", "genres", "split", "intent_group"))
            writer.writeheader()
            writer.writerows(rows)
        counts[split] = len(rows)
    return counts


if __name__ == "__main__":
    print(write_corpus())
