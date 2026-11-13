"""Evidence-based multi-label music categories for catalog metadata.

The iTunes Search API does not return lyrics or audio features. This classifier
uses verified release dates and provider genre metadata when available, and
uses optional licensed lyrics or audio features only when a caller supplies
them. It deliberately leaves unsupported labels off a track.
"""

from __future__ import annotations

import math
import re
from datetime import date
from typing import Any, Mapping


CATEGORIES = (
    {"id": "mass", "label": "Mass Songs", "description": "High-impact regional film tracks supported by metadata or audio evidence."},
    {"id": "party", "label": "Party Songs", "description": "Dance-oriented songs supported by genre or audio evidence."},
    {"id": "slow", "label": "Slow Songs", "description": "Slow-tempo or ballad-style tracks supported by genre or audio evidence."},
    {"id": "old", "label": "Old Songs", "description": "Tracks with release dates at least 25 years old."},
    {"id": "latest", "label": "Latest Songs", "description": "Tracks released within the last two years."},
    {"id": "romantic", "label": "Romantic Songs", "description": "Romantic metadata or emotional/lyrical cues from multiple sources."},
    {"id": "sad", "label": "Sad Songs", "description": "Sad metadata or emotional/lyrical cues from multiple sources."},
)
CATEGORY_IDS = {item["id"] for item in CATEGORIES}
CATEGORY_LABELS = {item["id"]: item["label"] for item in CATEGORIES}
CATEGORY_SEARCH_TERMS = {
    "mass": "mass songs",
    "party": "party dance songs",
    "slow": "slow ballad songs",
    "old": "old classic songs",
    "latest": "latest new songs",
    "romantic": "romantic love songs",
    "sad": "sad songs",
}

_MOOD_CATEGORY_WEIGHTS = {
    "happy": {"party": 0.9}, "sad": {"sad": 1.0, "slow": 0.35},
    "calm": {"slow": 0.9}, "energetic": {"mass": 0.8, "party": 0.9},
    "stressed": {"slow": 0.65}, "romantic": {"romantic": 1.0, "slow": 0.35},
    "neutral": {}, "motivated": {"mass": 0.8, "party": 0.65},
    "nostalgic": {"old": 0.85},
}
_GOAL_CATEGORY_WEIGHTS = {
    "relax": {"slow": 0.55}, "focus": {"slow": 0.2},
    "energize": {"mass": 0.6, "party": 0.65},
    "feel-good": {"party": 0.6}, "wind down": {"slow": 0.55},
    "discover something new": {"latest": 0.35},
}


def infer_music_categories(mood: str, goal: str = "") -> list[str]:
    """Return category preferences from both user choices, never song labels.

    The returned IDs guide relevance ranking. Track labels are still assigned
    separately by ``classify_music_categories`` from that track's evidence.
    """
    scores: dict[str, float] = {}
    for source in (_MOOD_CATEGORY_WEIGHTS.get(str(mood or "").casefold(), {}),
                   _GOAL_CATEGORY_WEIGHTS.get(str(goal or "").casefold(), {})):
        for category, weight in source.items():
            scores[category] = scores.get(category, 0.0) + weight
    order = {item["id"]: index for index, item in enumerate(CATEGORIES)}
    return sorted((category for category, score in scores.items() if score >= 0.35),
                  key=lambda category: (-scores[category], order[category]))

_ROMANTIC_CUES = (
    "romantic", "romance", "love", "lover", "valentine", "kiss", "darling",
    "ishq", "pyaar", "pyar", "prem", "prema", "kadhal", "kaadhal", "anbu",
)
_SAD_CUES = (
    "sad", "heartbreak", "heartbroken", "lonely", "alone", "cry", "tears",
    "pain", "broken", "grief", "dard", "udaas", "dukhi", "bewafa", "viraha", "biraha",
)
_DANCE_GENRES = ("dance", "disco", "edm", "house", "club", "party")
_SLOW_GENRES = ("ballad", "slow", "lullaby")
_INDIAN_FILM_GENRES = ("indian", "telugu", "tamil", "soundtrack", "bollywood", "tollywood", "regional")


def normalize_category(value: Any) -> str | None:
    """Accept category IDs or display labels and return the canonical ID."""
    if not isinstance(value, str) or not value.strip():
        return None
    folded = value.strip().casefold()
    for item in CATEGORIES:
        if folded in {item["id"], item["label"].casefold()}:
            return item["id"]
    return None


def _text(item: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_release_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        try:
            return date(int(raw[:4]), 1, 1) if re.fullmatch(r"\d{4}", raw) else None
        except ValueError:
            return None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _contains(text: str, terms: tuple[str, ...]) -> list[str]:
    folded = text.casefold()
    return [term for term in terms if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded)]


def classify_music_categories(
    item: Mapping[str, Any],
    *,
    audio_features: Mapping[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Return multi-label categories and the evidence supporting each label.

    Titles can contribute to emotional cues, but never determine a category on
    their own. Tempo/energy/danceability labels require the indicated signals
    together. No release date is inferred from title text.
    """
    today = today or date.today()
    genre = _text(item, "primaryGenreName", "genre", "track_genre")
    title = _text(item, "trackName", "song_name", "title")
    album = _text(item, "collectionName", "album_name")
    description = _text(item, "description")
    lyrics = _text(item, "lyrics", "licensed_lyrics")
    genre_folded = genre.casefold()
    album_folded = album.casefold()
    text_fields = {"title": title, "album": album, "genre": genre, "description": description}
    features = dict(item.get("audio_features") or {}) if isinstance(item.get("audio_features"), Mapping) else {}
    if isinstance(audio_features, Mapping):
        features.update(audio_features)
    for name in ("tempo", "energy", "danceability", "acousticness", "loudness"):
        if name not in features and name in item:
            features[name] = item[name]
    tempo = _number(features.get("tempo"))
    energy = _number(features.get("energy"))
    danceability = _number(features.get("danceability"))
    acousticness = _number(features.get("acousticness"))
    loudness = _number(features.get("loudness"))

    evidence: dict[str, list[str]] = {}
    scores: dict[str, float] = {}

    def add(category: str, score: float, *reasons: str) -> None:
        evidence[category] = list(reasons)
        scores[category] = round(max(0.0, min(1.0, score)), 2)

    release = _parse_release_date(item.get("releaseDate") or item.get("release_date"))
    try:
        old_cutoff = today.replace(year=today.year - 25)
    except ValueError:
        old_cutoff = today.replace(year=today.year - 25, day=28)
    if release and release <= old_cutoff:
        add("old", 1.0, f"catalog release date is {release.isoformat()} (25+ years old)")
    try:
        latest_cutoff = today.replace(year=today.year - 2)
    except ValueError:  # Leap day: use the last valid day of February.
        latest_cutoff = today.replace(year=today.year - 2, day=28)
    if release and latest_cutoff <= release <= today:
        add("latest", 1.0, f"catalog release date is {release.isoformat()} (within two years)")

    explicit_mass_genre = bool(re.search(r"(?<!\w)mass(?!\w)", genre_folded))
    mass_collection = bool(re.search(r"(?<!\w)mass(?!\w)", album_folded))
    indian_film_context = any(term in genre_folded for term in _INDIAN_FILM_GENRES)
    if explicit_mass_genre:
        add("mass", 0.95, f"provider genre explicitly identifies {genre}")
    elif (mass_collection and indian_film_context
          and energy is not None and energy >= 0.78
          and danceability is not None and danceability >= 0.65
          and tempo is not None and tempo >= 100):
        add("mass", 0.88, "album metadata identifies mass music", "high energy, danceability, and tempo support the label", f"provider genre is {genre}")

    dance_genre_hits = [term for term in _DANCE_GENRES if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", genre_folded)]
    if dance_genre_hits:
        add("party", 0.9, f"provider genre includes {', '.join(dance_genre_hits)}")
    elif (energy is not None and energy >= 0.65
          and danceability is not None and danceability >= 0.72
          and tempo is not None and tempo >= 100):
        add("party", 0.86, "audio energy, danceability, and tempo all support party suitability")

    slow_genre_hits = [term for term in _SLOW_GENRES if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", genre_folded)]
    if slow_genre_hits:
        add("slow", 0.88, f"provider genre includes {', '.join(slow_genre_hits)}")
    elif tempo is not None and energy is not None and tempo <= 85 and energy <= 0.55:
        add("slow", 0.9, f"audio tempo is {tempo:g} BPM", f"audio energy is {energy:g}")
    elif (tempo is not None and energy is not None and acousticness is not None
          and tempo <= 95 and energy <= 0.4 and acousticness >= 0.45):
        add("slow", 0.82, f"audio tempo is {tempo:g} BPM", f"low energy ({energy:g}) and acoustic character ({acousticness:g}) support a slow arrangement")

    for category, cues, display in (
        ("romantic", _ROMANTIC_CUES, "romantic"),
        ("sad", _SAD_CUES, "sad"),
    ):
        # A genre or album string alone is not enough to establish an emotion.
        # Emotional labels need multiple independent metadata cues or actual
        # lyric evidence, and song titles alone never establish them.
        field_matches = {name: _contains(text, cues) for name, text in text_fields.items() if text}
        matching_fields = [(name, terms) for name, terms in field_matches.items() if terms]
        lyric_matches = _contains(lyrics, cues) if lyrics else []
        distinct_cues = {term for _, terms in matching_fields for term in terms} | set(lyric_matches)
        if len(lyric_matches) >= 2:
            add(category, 0.9, f"available lyric text contains multiple {display} cues")
        elif len(matching_fields) >= 2 and len(distinct_cues) >= 2:
            add(category, 0.78, f"multiple catalog metadata fields contain {display} cues")

    category_status = {item["id"]: "assigned" if item["id"] in evidence else "unknown" for item in CATEGORIES}
    category_confidence = {item["id"]: scores.get(item["id"], 0.0) for item in CATEGORIES}
    return {
        "music_categories": [item["label"] for item in CATEGORIES if item["id"] in evidence],
        "category_ids": [item["id"] for item in CATEGORIES if item["id"] in evidence],
        "category_scores": scores,
        "category_confidence": category_confidence,
        "category_status": category_status,
        "category_evidence": evidence,
        "category_explanation": "; ".join(
            f"{next(entry['label'] for entry in CATEGORIES if entry['id'] == category)}: {', '.join(reasons)}"
            for category, reasons in evidence.items()
        ) or "No category label met the available metadata evidence threshold.",
    }
