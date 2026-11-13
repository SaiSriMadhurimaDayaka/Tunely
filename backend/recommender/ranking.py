"""Explainable feature matching and context-seeded exploration over Tunely's catalog.

Mood and listening goal are separate context fields. When the catalog has real
audio features, explicit targets for both can influence ranking; mood targets
are transparent feature proxies, not detected song labels or diagnoses. With
missing features the engine explores the catalog without popularity ranking.
No learned personalization is claimed here: Q-Learning is a separate layer.
"""

from __future__ import annotations

import math
import hashlib
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence
from .external_links import listening_links


DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "tunely.sqlite3"
FEATURES = (
    "danceability", "energy", "acousticness", "instrumentalness", "valence",
    "speechiness", "liveness", "tempo", "loudness",
)
MOODS = {"calm", "happy", "sad", "stressed", "energetic", "neutral", "romantic", "motivated", "nostalgic"}
MOOD_TARGETS: dict[str, dict[str, float]] = {
    # These are transparent energy/valence ranking proxies, never inferred labels.
    "calm": {"energy": 0.28, "valence": 0.55, "acousticness": 0.62},
    "happy": {"energy": 0.66, "valence": 0.82, "danceability": 0.66},
    "sad": {"energy": 0.30, "valence": 0.28, "acousticness": 0.56},
    "stressed": {"energy": 0.28, "valence": 0.48, "acousticness": 0.62},
    "energetic": {"energy": 0.84, "valence": 0.70, "danceability": 0.74},
    "neutral": {"energy": 0.50, "valence": 0.52},
    "romantic": {"energy": 0.42, "valence": 0.66, "acousticness": 0.52},
    "motivated": {"energy": 0.76, "valence": 0.70, "danceability": 0.60},
    "nostalgic": {"energy": 0.42, "valence": 0.52, "acousticness": 0.54},
}
GOAL_TARGETS: dict[str, dict[str, float]] = {
    "focus": {"energy": 0.40, "instrumentalness": 0.55, "speechiness": 0.12, "danceability": 0.35},
    "focus and study": {"energy": 0.40, "instrumentalness": 0.55, "speechiness": 0.12, "danceability": 0.35},
    "relax": {"energy": 0.28, "acousticness": 0.68, "valence": 0.48, "liveness": 0.15},
    "relax and unwind": {"energy": 0.28, "acousticness": 0.68, "valence": 0.48, "liveness": 0.15},
    "energize": {"energy": 0.82, "danceability": 0.76, "valence": 0.70, "tempo": 125.0},
    "find energetic music": {"energy": 0.82, "danceability": 0.76, "valence": 0.70, "tempo": 125.0},
    "feel-good": {"energy": 0.6, "valence": 0.8},
    "wind down": {"energy": 0.22, "acousticness": 0.62, "valence": 0.40, "liveness": 0.12},
    "discover something new": {},
}
MOOD_LABELS = {value: value.title() for value in MOODS}
GOAL_LABELS = {
    "focus": "Focus", "focus and study": "Focus and study", "relax": "Relax", "relax and unwind": "Relax and unwind",
    "energize": "Energize", "find energetic music": "Find energetic music", "feel-good": "Feel-good", "wind down": "Wind down",
    "discover something new": "Discover something new",
}
SCALES = {"tempo": (0.0, 250.0), "loudness": (-60.0, 0.0)}


class RecommendationError(ValueError):
    """An invalid or unsupported recommendation request."""


def _normalized_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split()) if value is not None else ""


def _feature_value(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def targets_for_context(context: Mapping[str, Any]) -> dict[str, float]:
    """Build explainable feature targets, scaling mood influence by intensity.

    Mild uses a smaller shift from a neutral midpoint; moderate uses 70% of
    the mood proxy and intense uses its full target. A goal is blended in
    separately, so it can meaningfully refine the mood rather than replace it.
    """
    mood = _normalized_text(context.get("mood"))
    goal = _normalized_text(context.get("goal"))
    level = context.get("mood_level", 2)
    if not isinstance(level, int) or isinstance(level, bool) or level not in (1, 2, 3):
        raise RecommendationError("mood_level must be 1 (mild), 2 (moderate), or 3 (intense).")
    targets = dict(GOAL_TARGETS.get(goal, {}))
    intensity = {1: 0.35, 2: 0.70, 3: 1.0}[level]
    neutral = {"energy": 0.5, "valence": 0.5, "danceability": 0.5, "acousticness": 0.5,
               "instrumentalness": 0.5, "speechiness": 0.5, "liveness": 0.5, "tempo": 100.0,
               "loudness": -30.0}
    for feature, target in MOOD_TARGETS.get(mood, {}).items():
        mood_target = neutral[feature] + (target - neutral[feature]) * intensity
        targets[feature] = (targets[feature] + mood_target) / 2 if feature in targets else mood_target
    return targets


class RecommendationEngine:
    """Read a SQLite song catalog and produce ranked, explained suggestions."""

    def __init__(self, database: str | Path = DEFAULT_DB):
        self.database = Path(database)

    def _load_songs(self) -> list[dict[str, Any]]:
        if not self.database.is_file():
            raise RecommendationError(f"Song catalog not found: {self.database}")
        try:
            with closing(sqlite3.connect(self.database)) as conn, conn:
                conn.row_factory = sqlite3.Row
                table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='songs'").fetchone()
                if table is None:
                    raise RecommendationError("The database has no songs table. Run data/prepare_dataset.py first.")
                return [dict(row) for row in conn.execute("SELECT * FROM songs")]
        except sqlite3.Error as exc:
            raise RecommendationError("Could not read the song catalog.") from exc

    def recommend(
        self,
        context: Mapping[str, Any] | None = None,
        *,
        limit: int = 12,
        strategy: str = "auto",
        exclude_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Return recommendations and metadata.

        Supported context keys: mood, goal, genre, language, explicit,
        feature_preferences (feature -> target value). Mood/goal targets use
        available audio features only. Supported strategies:
        ``auto``, ``features``, and ``popularity``. Feature targets for audio
        features in [0, 1] are used directly; tempo is in BPM and loudness in dB.
        """
        context = dict(context or {})
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise RecommendationError("limit must be an integer between 1 and 100.")
        strategy = _normalized_text(strategy) or "auto"
        if strategy not in {"auto", "features", "popularity", "exploration"}:
            raise RecommendationError("strategy must be 'auto', 'features', 'exploration', or 'popularity'.")
        mood = _normalized_text(context.get("mood"))
        goal = _normalized_text(context.get("goal"))
        if mood and mood not in MOODS:
            raise RecommendationError(f"Unsupported mood. Choose one of: {', '.join(sorted(MOODS))}.")
        if goal and goal not in GOAL_TARGETS:
            raise RecommendationError(f"Unsupported listening goal. Choose one of: {', '.join(GOAL_TARGETS)}.")
        preferences = context.get("feature_preferences") or {}
        if not isinstance(preferences, Mapping):
            raise RecommendationError("feature_preferences must be an object of feature names and target values.")
        targets = targets_for_context(context)
        for name, raw_target in preferences.items():
            feature = _normalized_text(name)
            if feature not in FEATURES:
                raise RecommendationError(f"Unsupported feature '{name}'. Supported features: {', '.join(FEATURES)}.")
            target = _feature_value(raw_target)
            if target is None:
                raise RecommendationError(f"Target for {feature} must be a finite number.")
            low, high = SCALES.get(feature, (0.0, 1.0))
            if not low <= target <= high:
                raise RecommendationError(f"Target for {feature} must be between {low:g} and {high:g}.")
            targets[feature] = target

        songs = self._load_songs()
        excluded = {str(item) for item in exclude_ids}
        songs = [song for song in songs if str(song.get("song_id")) not in excluded and str(song.get("track_id")) not in excluded]
        genre = _normalized_text(context.get("genre"))
        if genre:
            songs = [song for song in songs if _normalized_text(song.get("track_genre")) == genre]
        language = _normalized_text(context.get("language"))
        if language:
            if language == "all languages":
                language = ""
            elif language == "unknown":
                songs = [song for song in songs if not _normalized_text(song.get("language"))]
            else:
                songs = [song for song in songs if language in {_normalized_text(label) for label in re.split(r"[,;|]", str(song.get("language") or "")) if label.strip()}]
        if "explicit" in context and context["explicit"] is not None:
            if not isinstance(context["explicit"], bool):
                raise RecommendationError("explicit must be true or false.")
            songs = [song for song in songs if song.get("explicit") is not None and bool(song["explicit"]) == context["explicit"]]
        if not songs:
            message = (f"No songs are tagged as {str(context.get('language')).strip()}. Try another language or All Languages."
                       if context.get("language") and _normalized_text(context.get("language")) not in {"unknown", "all languages"}
                       else "No songs match the selected filters. Try changing your choices or clearing a filter.")
            return {"recommendations": [], "strategy": "none", "context": self._public_context(mood, goal, context.get("mood_level", 2)), "message": message}

        feature_strategy = strategy == "features" or (strategy == "auto" and bool(targets))
        if feature_strategy and not targets:
            raise RecommendationError("Feature ranking needs a listening goal or at least one feature preference.")
        ranked: list[tuple[float, dict[str, Any], list[str]]] = []
        actual_strategy = "features" if feature_strategy else ("popularity" if strategy == "popularity" else "exploration")
        for song in songs:
            if feature_strategy:
                score, matched = self._feature_score(song, targets)
                if not matched:
                    continue
            elif actual_strategy == "exploration":
                key = "|".join((mood or "", str(context.get("mood_level", 2)), goal or "", str(song.get("song_id") or "")))
                score = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / (2**64 - 1)
                matched = []
            else:
                score = float(song.get("popularity") or 0) / 100.0
                matched = []
            ranked.append((score, song, matched))
        if feature_strategy and not ranked:
            if strategy == "features":
                raise RecommendationError("None of the filtered songs has the requested audio features.")
            actual_strategy = "exploration"
            ranked = []
            for song in songs:
                key = "|".join((mood or "", str(context.get("mood_level", 2)), goal or "", str(song.get("song_id") or "")))
                score = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / (2**64 - 1)
                ranked.append((score, song, []))
        if actual_strategy == "exploration":
            ranked.sort(key=lambda item: (item[0], str(item[1].get("song_id") or "")), reverse=True)
        else:
            ranked.sort(key=lambda item: (item[0], int(item[1].get("popularity") or 0), str(item[1].get("song_id") or "")), reverse=True)
        results = [self._format_song(song, score, matched, actual_strategy, mood, goal, targets,
                                     context.get("mood_level", 2)) for score, song, matched in ranked[:limit]]
        return {
            "recommendations": results,
            "strategy": actual_strategy,
            "catalog_size": len(songs),
            "returned": len(results),
            "context": self._public_context(mood, goal, context.get("mood_level", 2)),
            "personalization": {"status": "not_yet_trained", "message": "Recommendations use available catalog features or context-seeded exploration; popularity ranking is available only when explicitly requested. No user feedback has trained a model."},
        }

    @staticmethod
    def _feature_score(song: Mapping[str, Any], targets: Mapping[str, float]) -> tuple[float, list[str]]:
        weighted_errors: list[tuple[float, float]] = []
        matched: list[str] = []
        for feature, target in targets.items():
            value = _feature_value(song.get(feature))
            if value is None:
                continue
            low, high = SCALES.get(feature, (0.0, 1.0))
            distance = min(1.0, abs(value - target) / (high - low))
            weight = 1.0
            weighted_errors.append((distance, weight))
            matched.append(feature)
        if not weighted_errors:
            return 0.0, []
        # Mean normalized absolute error over only the requested features present on this track.
        error = sum(distance * weight for distance, weight in weighted_errors) / sum(weight for _, weight in weighted_errors)
        return 1.0 - error, matched

    @staticmethod
    def _format_song(song: Mapping[str, Any], score: float, matched: list[str], strategy: str,
                     mood: str, goal: str, targets: Mapping[str, float], mood_level: int = 2) -> dict[str, Any]:
        characteristics = {feature: song.get(feature) for feature in FEATURES if song.get(feature) is not None}
        if strategy == "features":
            best = sorted(matched, key=lambda f: abs(float(song[f]) - targets[f]) / (SCALES.get(f, (0.0, 1.0))[1] - SCALES.get(f, (0.0, 1.0))[0]))[:3]
            reason = "Closest match on " + ", ".join(best) + "."
            if goal:
                reason = f"Ranked for {GOAL_LABELS[goal].lower()}; " + reason[0].lower() + reason[1:]
            if mood:
                intensity_label = {1: "mild", 2: "moderate", 3: "intense"}[mood_level]
                reason += f" Mood fit uses your {intensity_label} {MOOD_LABELS[mood].lower()} preference and available audio features; it is not a detected song mood."
        elif strategy == "exploration":
            intensity_label = {1: "mild", 2: "moderate", 3: "intense"}[mood_level]
            reason = f"Catalog exploration pick for context only ({intensity_label} intensity). This catalog has no verified song mood tags or audio features, so mood fit cannot be ranked."
        else:
            reason = "Among the most popular tracks in the filtered catalog."
        if mood:
            reason += f" Your selected mood ({MOOD_LABELS[mood]}) is included as context; it does not determine your listening goal."
        return {
            "song_id": song.get("song_id"), "track_id": song.get("track_id"),
            "song_name": song.get("song_name"), "artist": song.get("artist"),
            **listening_links(song.get("song_name"), song.get("artist"), song.get("youtube_music_url")),
            "language": song.get("language"), "mood": song.get("mood"),
            "album_name": song.get("album_name"), "release_year": song.get("release_year"),
            "popularity": song.get("popularity"), "duration_ms": song.get("duration_ms"),
            "duration_minutes": song.get("duration_minutes"), "album_type": song.get("album_type"),
            "cover_image": song.get("cover_image"), "genre": song.get("track_genre"),
            "characteristics": characteristics, "score": round(score, 5),
            "score_method": "one minus mean normalized absolute error across available goal and mood proxy features" if strategy == "features" else ("context-seeded catalog exploration" if strategy == "exploration" else "popularity divided by 100"),
            "matched_features": matched, "explanation": reason,
            "listening_url": None,
        }

    @staticmethod
    def _public_context(mood: str, goal: str, mood_level: int = 2) -> dict[str, Any]:
        return {"mood": MOOD_LABELS.get(mood), "mood_level": mood_level, "goal": GOAL_LABELS.get(goal)}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Preview Tunely recommendations from the local catalog.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--mood", choices=sorted(MOODS), default="neutral")
    parser.add_argument("--goal", choices=list(GOAL_TARGETS), default="discover something new")
    parser.add_argument("--genre")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    output = RecommendationEngine(args.database).recommend({"mood": args.mood, "goal": args.goal, "genre": args.genre}, limit=args.limit)
    print(json.dumps(output, indent=2))
