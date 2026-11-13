"""Content-based ranking over imported Saraga annotations (never popularity)."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence
from .external_links import listening_links

from .ranking import RecommendationError, _normalized_text


SARAGA_GOALS = {
    "relax", "focus", "energize", "feel-good", "wind down", "discover something new",
    "relax and unwind", "focus and study", "find energetic music",
}
SARAGA_MOODS = {"calm", "happy", "sad", "low", "stressed", "energetic", "neutral"}
TEXT_FIELDS = ("raga", "tala", "laya", "form", "work")
PREFERENCE_FIELDS = (*TEXT_FIELDS, "tempo_bpm")


class SaragaRecommendationError(RecommendationError):
    """Invalid or unavailable Saraga recommendation input."""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the additive Saraga table and indexes without touching Spotify."""
    conn.execute("""CREATE TABLE IF NOT EXISTS recordings (
        recording_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        artist TEXT,
        tradition TEXT NOT NULL CHECK(tradition IN ('hindustani', 'carnatic')),
        release_info TEXT,
        raga TEXT,
        tala TEXT,
        laya TEXT,
        form TEXT,
        work TEXT,
        tempo_bpm REAL,
        tempo_annotations_json TEXT,
        duration_ms INTEGER CHECK(duration_ms IS NULL OR duration_ms > 0),
        language TEXT,
        mood TEXT,
        youtube_music_url TEXT,
        audio_path TEXT,
        audio_status TEXT NOT NULL CHECK(audio_status IN ('not_downloaded', 'available_local')),
        dataset_source TEXT NOT NULL,
        dataset_license TEXT NOT NULL,
        attribution TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        source_path TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )""")
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(recordings)")}
    for column in ("mood", "youtube_music_url"):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE recordings ADD COLUMN {column} TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_tradition ON recordings(tradition)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_raga ON recordings(raga)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_tala ON recordings(tala)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_recordings_language ON recordings(language)")


class SaragaRecommendationEngine:
    """Rank by exact user-selected annotations and rotate unseen recordings."""

    def __init__(self, database: str | Path):
        self.database = Path(database)

    def recommend(
        self,
        context: Mapping[str, Any] | None = None,
        *,
        limit: int = 8,
        exclude_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        context = dict(context or {})
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise SaragaRecommendationError("limit must be an integer between 1 and 100.")

        mood = _normalized_text(context.get("mood"))
        goal = _normalized_text(context.get("goal"))
        if mood and mood not in SARAGA_MOODS:
            raise SaragaRecommendationError("Choose a supported mood: calm, happy, sad, stressed, energetic, or neutral.")
        if goal and goal not in SARAGA_GOALS:
            raise SaragaRecommendationError("Choose a supported listening goal: relax, focus, energize, feel-good, wind down, or discover something new.")
        level = context.get("mood_level", 3)
        if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 5:
            raise SaragaRecommendationError("mood_level must be an integer from 1 to 5.")

        tradition = _normalized_text(context.get("tradition"))
        if tradition and tradition not in {"hindustani", "carnatic"}:
            raise SaragaRecommendationError("tradition must be 'hindustani' or 'carnatic'.")
        language = _normalized_text(context.get("language"))
        preferences = context.get("preferences") or {}
        if not isinstance(preferences, Mapping):
            raise SaragaRecommendationError("preferences must be an object with raga, tala, laya, form, work, or tempo_bpm.")
        normalized_preferences: dict[str, Any] = {}
        for raw_field, raw_value in preferences.items():
            field = _normalized_text(raw_field)
            if field not in PREFERENCE_FIELDS:
                raise SaragaRecommendationError(f"Unsupported Saraga preference '{raw_field}'.")
            if field == "tempo_bpm":
                try:
                    target = float(raw_value)
                except (TypeError, ValueError) as exc:
                    raise SaragaRecommendationError("tempo_bpm preference must be a number.") from exc
                if not math.isfinite(target) or not 1 <= target <= 500:
                    raise SaragaRecommendationError("tempo_bpm preference must be between 1 and 500.")
                normalized_preferences[field] = target
            elif not isinstance(raw_value, str) or not raw_value.strip():
                raise SaragaRecommendationError(f"{field} preference must be a non-empty string.")
            else:
                normalized_preferences[field] = _normalized_text(raw_value)

        if language:
            with closing(sqlite3.connect(self.database)) as conn, conn:
                languages = [row[0] for row in conn.execute(
                    "SELECT DISTINCT language FROM recordings WHERE language IS NOT NULL AND TRIM(language)<>''"
                )]
            if not languages:
                raise SaragaRecommendationError("Language filtering is unavailable: no Saraga recordings have verified language metadata.")
            if language not in {_normalized_text(item) for item in languages}:
                raise SaragaRecommendationError("No Saraga recordings have the selected language.")

        if not self.database.is_file():
            raise SaragaRecommendationError("The Tunely database is not available.")
        try:
            with closing(sqlite3.connect(self.database)) as conn, conn:
                conn.row_factory = sqlite3.Row
                rows = [dict(row) for row in conn.execute("SELECT * FROM recordings")]
        except sqlite3.Error as exc:
            raise SaragaRecommendationError("Could not read Saraga recordings.") from exc

        excluded = {str(value) for value in exclude_ids}
        candidates = [row for row in rows if str(row["recording_id"]) not in excluded]
        if tradition:
            candidates = [row for row in candidates if _normalized_text(row.get("tradition")) == tradition]
        if language:
            candidates = [row for row in candidates if _normalized_text(row.get("language")) == language]

        ranked: list[tuple[float, dict[str, Any], list[str]]] = []
        for row in candidates:
            score = 0.0
            matched: list[str] = []
            for field, target in normalized_preferences.items():
                if field == "tempo_bpm":
                    try:
                        actual = float(row.get("tempo_bpm"))
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(actual):
                        continue
                    closeness = max(0.0, 1.0 - abs(actual - target) / max(target, 60.0))
                    score += closeness
                    if closeness > 0:
                        matched.append(field)
                    continue
                actual = _normalized_text(row.get(field))
                if actual and target in actual:
                    score += 1.0
                    matched.append(field)
            ranked.append((score, row, matched))

        ranked.sort(key=lambda item: (item[0], _normalized_text(item[1].get("title")), str(item[1].get("recording_id"))), reverse=True)
        recommendations = [
            self._format(row, score, matched, mood, goal, level, bool(normalized_preferences))
            for score, row, matched in ranked[:limit]
        ]
        strategy = "metadata_match" if normalized_preferences else "catalog_exploration"
        return {
            "recommendations": recommendations,
            "strategy": strategy,
            "catalog_size": len(candidates),
            "returned": len(recommendations),
            "context": {"mood": mood or None, "mood_level": level, "goal": goal or None, "tradition": tradition or None, "language": language or None},
            "ranking_method": "Exact annotation matches for explicit raga/tala/laya/form/work preferences; normalized distance for available Carnatic BPM. No popularity or inferred mood-to-raga mapping is used.",
            "personalization": {"status": "not_yet_trained", "message": "Saraga recommendations use available annotations. Ratings are not yet sufficient to train a Saraga strategy policy."},
        }

    @staticmethod
    def _format(row: Mapping[str, Any], score: float, matched: list[str], mood: str, goal: str, level: int, used_preferences: bool) -> dict[str, Any]:
        if matched:
            reason = "Matches your selected " + ", ".join(matched) + " preference" + ("s." if len(matched) > 1 else ".")
        else:
            reason = "Exploration pick from the " + str(row.get("tradition") or "Saraga") + " collection."
        if used_preferences and not matched:
            reason += " The available annotations did not match your requested characteristics."
        if mood or goal:
            selected = []
            if mood:
                selected.append(f"mood {mood} (level {level})")
            if goal:
                selected.append(f"goal {goal}")
            reason += " Selected " + " and ".join(selected) + " are kept separate; Saraga does not provide a validated universal mood mapping."
        audio_available = row.get("audio_status") == "available_local" and bool(row.get("audio_path"))
        return {
            "recording_id": row.get("recording_id"), "title": row.get("title"), "artist": row.get("artist"),
            **listening_links(row.get("title"), row.get("artist"), row.get("youtube_music_url")),
            "mood": row.get("mood"),
            "tradition": row.get("tradition"), "release_info": row.get("release_info"),
            "raga": row.get("raga"), "tala": row.get("tala"), "laya": row.get("laya"),
            "form": row.get("form"), "work": row.get("work"), "tempo_bpm": row.get("tempo_bpm"),
            "tempo_annotations": json.loads(row["tempo_annotations_json"]) if row.get("tempo_annotations_json") else [],
            "duration_ms": row.get("duration_ms"), "language": row.get("language"),
            "audio_available": audio_available, "playback_url": f"/api/audio/{row['recording_id']}" if audio_available else None,
            "score": round(score, 5), "matched_fields": matched, "explanation": reason,
            "dataset_source": row.get("dataset_source"), "dataset_license": row.get("dataset_license"),
            "attribution": row.get("attribution"),
        }
