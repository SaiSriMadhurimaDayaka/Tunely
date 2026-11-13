"""Persistent, incrementally refreshed iTunes catalog dataset for Tunely."""

from __future__ import annotations

import json
import sqlite3
import unicodedata
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

from backend.recommender.itunes_discovery import _language_confidence
from backend.recommender.music_categories import CATEGORIES, classify_music_categories


def ensure_itunes_dataset_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS itunes_tracks (
            track_id TEXT PRIMARY KEY,
            track_name TEXT NOT NULL,
            track_name_normalized TEXT NOT NULL,
            artist_name TEXT,
            artist_normalized TEXT,
            collection_name TEXT,
            collection_normalized TEXT,
            primary_genre TEXT,
            genre_normalized TEXT,
            release_date TEXT,
            release_year INTEGER,
            track_time_millis INTEGER,
            artwork_url TEXT,
            itunes_track_url TEXT,
            preview_url TEXT,
            language TEXT,
            language_confidence REAL NOT NULL DEFAULT 0,
            language_evidence TEXT NOT NULL DEFAULT 'unknown',
            country TEXT,
            collection_id TEXT,
            category_ids_json TEXT NOT NULL DEFAULT '[]',
            category_scores_json TEXT NOT NULL DEFAULT '{}',
            category_confidence_json TEXT NOT NULL DEFAULT '{}',
            category_status_json TEXT NOT NULL DEFAULT '{}',
            category_evidence_json TEXT NOT NULL DEFAULT '{}',
            category_model_version TEXT NOT NULL DEFAULT 'legacy',
            lyrics_availability TEXT NOT NULL DEFAULT 'unknown',
            lyrics_lookup_status TEXT NOT NULL DEFAULT 'not_checked',
            audio_analysis_status TEXT NOT NULL DEFAULT 'unavailable_no_authorized_audio',
            emotion_labels_json TEXT NOT NULL DEFAULT '[]',
            emotion_confidence_json TEXT NOT NULL DEFAULT '{}',
            emotion_evidence_json TEXT NOT NULL DEFAULT '{}',
            emotion_model_version TEXT NOT NULL DEFAULT 'not_classified',
            emotion_confidence_kind TEXT NOT NULL DEFAULT 'unknown',
            raw_json TEXT NOT NULL,
            first_collected_at TEXT NOT NULL,
            last_collected_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_itunes_tracks_year ON itunes_tracks(release_year);
        CREATE INDEX IF NOT EXISTS idx_itunes_tracks_genre ON itunes_tracks(genre_normalized);
        CREATE INDEX IF NOT EXISTS idx_itunes_tracks_language ON itunes_tracks(language);
        CREATE TABLE IF NOT EXISTS itunes_track_languages (
            track_id TEXT NOT NULL REFERENCES itunes_tracks(track_id) ON DELETE CASCADE,
            requested_language TEXT NOT NULL,
            first_collected_at TEXT NOT NULL,
            last_collected_at TEXT NOT NULL,
            PRIMARY KEY(track_id, requested_language)
        );
        CREATE INDEX IF NOT EXISTS idx_itunes_track_languages_language
            ON itunes_track_languages(requested_language, last_collected_at DESC);
        CREATE TABLE IF NOT EXISTS itunes_collection_queries (
            query_key TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            requested_language TEXT NOT NULL,
            country TEXT NOT NULL,
            result_count INTEGER NOT NULL DEFAULT 0,
            last_collected_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS itunes_lyrics_cache (
            query_key TEXT PRIMARY KEY,
            lookup_status TEXT NOT NULL,
            lyrics_available INTEGER NOT NULL DEFAULT 0,
            lrclib_record_id TEXT,
            matched_title TEXT,
            matched_artist TEXT,
            match_score REAL,
            emotion_labels_json TEXT NOT NULL DEFAULT '[]',
            emotion_confidence_json TEXT NOT NULL DEFAULT '{}',
            emotion_evidence_json TEXT NOT NULL DEFAULT '{}',
            emotion_model_version TEXT NOT NULL DEFAULT 'not_classified',
            emotion_confidence_kind TEXT NOT NULL DEFAULT 'unknown',
            checked_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
    """)
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(itunes_tracks)")}
    for column, definition in (("category_confidence_json", "TEXT NOT NULL DEFAULT '{}'"),
                               ("category_status_json", "TEXT NOT NULL DEFAULT '{}'"),
                               ("category_model_version", "TEXT NOT NULL DEFAULT 'legacy'"),
                               ("lyrics_availability", "TEXT NOT NULL DEFAULT 'unknown'"),
                               ("lyrics_lookup_status", "TEXT NOT NULL DEFAULT 'not_checked'"),
                               ("audio_analysis_status", "TEXT NOT NULL DEFAULT 'unavailable_no_authorized_audio'"),
                               ("emotion_labels_json", "TEXT NOT NULL DEFAULT '[]'"),
                               ("emotion_confidence_json", "TEXT NOT NULL DEFAULT '{}'"),
                               ("emotion_evidence_json", "TEXT NOT NULL DEFAULT '{}'"),
                               ("emotion_model_version", "TEXT NOT NULL DEFAULT 'not_classified'"),
                               ("emotion_confidence_kind", "TEXT NOT NULL DEFAULT 'unknown'")):
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE itunes_tracks ADD COLUMN {column} {definition}")
    # Recompute stored labels once when evidence rules change. The tracks remain
    # real provider records; this migration only refreshes classifier output.
    for row in conn.execute("SELECT track_id,raw_json FROM itunes_tracks WHERE category_model_version<>'music-categories-v2'"):
        try:
            item = json.loads(row[1])
            classification = classify_music_categories(item)
        except (TypeError, json.JSONDecodeError):
            classification = classify_music_categories({})
        conn.execute("""UPDATE itunes_tracks SET category_ids_json=?,category_scores_json=?,
            category_confidence_json=?,category_status_json=?,category_evidence_json=?,
            category_model_version='music-categories-v2' WHERE track_id=?""",
            (json.dumps(classification["category_ids"], ensure_ascii=False),
             json.dumps(classification["category_scores"], ensure_ascii=False),
             json.dumps(classification["category_confidence"], ensure_ascii=False),
             json.dumps(classification["category_status"], ensure_ascii=False),
             json.dumps(classification["category_evidence"], ensure_ascii=False), row[0]))


def _normalized(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    collapsed = " ".join(unicodedata.normalize("NFKC", value).split())
    return collapsed.casefold() or None


def _https_url(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value.startswith("http://"):
        value = "https://" + value[len("http://"):]
    return value if value.startswith("https://") else None


def _track_values(item: Mapping[str, Any], requested_language: str, country: str) -> dict[str, Any] | None:
    if item.get("kind") != "song":
        return None
    track_id = item.get("trackId")
    title = item.get("trackName")
    if not isinstance(track_id, (str, int)) or not str(track_id).isdigit():
        return None
    if not isinstance(title, str) or not title.strip():
        return None
    title = " ".join(title.split())
    artist = item.get("artistName")
    artist = " ".join(artist.split()) if isinstance(artist, str) and artist.strip() else None
    album = item.get("collectionName")
    album = " ".join(album.split()) if isinstance(album, str) and album.strip() else None
    genre = item.get("primaryGenreName")
    genre = " ".join(genre.split()) if isinstance(genre, str) and genre.strip() else None
    raw_release = item.get("releaseDate")
    release_date = release_year = None
    if isinstance(raw_release, str):
        try:
            parsed = date.fromisoformat(raw_release[:10])
            release_date, release_year = parsed.isoformat(), parsed.year
        except ValueError:
            pass
    duration = item.get("trackTimeMillis")
    if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
        duration = None
    evidence_text = " ".join(str(item.get(key) or "") for key in
                             ("trackName", "artistName", "collectionName", "primaryGenreName"))
    language_confidence, language_evidence = _language_confidence(evidence_text, requested_language)
    # Only explicit or script-specific evidence is stored as a detected language.
    detected_language = requested_language if language_confidence >= 0.90 else None
    classification = classify_music_categories(item)
    artwork = _https_url(item.get("artworkUrl100") or item.get("artworkUrl60"))
    return {
        "track_id": str(track_id), "track_name": title, "track_name_normalized": _normalized(title),
        "artist_name": artist, "artist_normalized": _normalized(artist),
        "collection_name": album, "collection_normalized": _normalized(album),
        "primary_genre": genre, "genre_normalized": _normalized(genre),
        "release_date": release_date, "release_year": release_year,
        "track_time_millis": duration, "artwork_url": artwork,
        "itunes_track_url": _https_url(item.get("trackViewUrl")),
        "preview_url": _https_url(item.get("previewUrl")),
        "language": detected_language, "language_confidence": language_confidence,
        "language_evidence": language_evidence, "country": country.strip().lower(),
        "collection_id": str(item["collectionId"]) if str(item.get("collectionId", "")).isdigit() else None,
        "category_ids_json": json.dumps(classification["category_ids"], ensure_ascii=False),
        "category_scores_json": json.dumps(classification["category_scores"], ensure_ascii=False),
        "category_confidence_json": json.dumps(classification["category_confidence"], ensure_ascii=False),
        "category_status_json": json.dumps(classification["category_status"], ensure_ascii=False),
        "category_evidence_json": json.dumps(classification["category_evidence"], ensure_ascii=False),
        "category_model_version": "music-categories-v2",
        "raw_json": json.dumps(dict(item), ensure_ascii=False),
        "requested_language": requested_language,
    }


_TRACK_COLUMNS = (
    "track_name", "track_name_normalized", "artist_name", "artist_normalized",
    "collection_name", "collection_normalized", "primary_genre", "genre_normalized",
    "release_date", "release_year", "track_time_millis", "artwork_url", "itunes_track_url",
    "preview_url", "language", "language_confidence", "language_evidence", "country",
    "collection_id", "category_ids_json", "category_scores_json", "category_confidence_json",
    "category_status_json", "category_evidence_json", "category_model_version", "raw_json",
)


def upsert_itunes_tracks(conn: sqlite3.Connection, items: Sequence[Mapping[str, Any]], *,
                         requested_language: str, country: str) -> dict[str, int]:
    """Add/update real provider records without deleting existing metadata."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    inserted = updated = invalid = 0
    seen: set[str] = set()
    for item in items:
        values = _track_values(item, requested_language, country)
        if not values:
            invalid += 1
            continue
        track_id = values["track_id"]
        if track_id in seen:
            continue
        seen.add(track_id)
        existing = conn.execute("SELECT raw_json FROM itunes_tracks WHERE track_id=?", (track_id,)).fetchone()
        if existing:
            try:
                merged_item = json.loads(existing["raw_json"])
            except (TypeError, json.JSONDecodeError):
                merged_item = {}
            for field, value in item.items():
                if value is not None and value != "":
                    merged_item[field] = value
            values = _track_values(merged_item, requested_language, country) or values
            preserved_columns = set(("language", "language_confidence", "language_evidence",
                                     "category_ids_json", "category_scores_json", "category_confidence_json",
                                     "category_status_json", "category_evidence_json"))
            assignments = ",".join(f"{column}=COALESCE(excluded.{column},itunes_tracks.{column})"
                                   for column in _TRACK_COLUMNS if column not in preserved_columns)
            conn.execute(f"INSERT INTO itunes_tracks(track_id,{','.join(_TRACK_COLUMNS)},first_collected_at,last_collected_at) "
                         f"VALUES({','.join('?' for _ in range(len(_TRACK_COLUMNS)+3))}) "
                         f"ON CONFLICT(track_id) DO UPDATE SET {assignments}, "
                         "language=CASE WHEN excluded.language_confidence>itunes_tracks.language_confidence "
                         "THEN excluded.language ELSE itunes_tracks.language END, "
                         "language_confidence=MAX(itunes_tracks.language_confidence,excluded.language_confidence), "
                         "language_evidence=CASE WHEN excluded.language_confidence>itunes_tracks.language_confidence "
                         "THEN excluded.language_evidence ELSE itunes_tracks.language_evidence END, "
                         "category_ids_json=CASE WHEN excluded.category_ids_json<>'[]' THEN excluded.category_ids_json ELSE itunes_tracks.category_ids_json END, "
                         "category_scores_json=CASE WHEN excluded.category_scores_json<>'{}' THEN excluded.category_scores_json ELSE itunes_tracks.category_scores_json END, "
                         "category_confidence_json=CASE WHEN excluded.category_confidence_json<>'{}' THEN excluded.category_confidence_json ELSE itunes_tracks.category_confidence_json END, "
                         "category_status_json=CASE WHEN excluded.category_status_json<>'{}' THEN excluded.category_status_json ELSE itunes_tracks.category_status_json END, "
                         "category_evidence_json=CASE WHEN excluded.category_evidence_json<>'{}' THEN excluded.category_evidence_json ELSE itunes_tracks.category_evidence_json END, "
                         "last_collected_at=excluded.last_collected_at",
                         (track_id, *(values[column] for column in _TRACK_COLUMNS), now, now))
            updated += 1
        else:
            conn.execute(f"INSERT INTO itunes_tracks(track_id,{','.join(_TRACK_COLUMNS)},first_collected_at,last_collected_at) "
                         f"VALUES({','.join('?' for _ in range(len(_TRACK_COLUMNS)+3))})",
                         (track_id, *(values[column] for column in _TRACK_COLUMNS), now, now))
            inserted += 1
        conn.execute("INSERT INTO itunes_track_languages(track_id,requested_language,first_collected_at,last_collected_at) "
                     "VALUES(?,?,?,?) ON CONFLICT(track_id,requested_language) DO UPDATE SET last_collected_at=excluded.last_collected_at",
                     (track_id, requested_language, now, now))
    return {"inserted": inserted, "updated": updated, "invalid": invalid}


def local_itunes_items(conn: sqlite3.Connection, language: str, limit: int = 500) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT t.raw_json,t.lyrics_availability,t.lyrics_lookup_status,t.audio_analysis_status,"
                        "t.emotion_labels_json,t.emotion_confidence_json,t.emotion_evidence_json,"
                        "t.emotion_model_version,t.emotion_confidence_kind FROM itunes_tracks t JOIN itunes_track_languages l USING(track_id) "
                        "WHERE l.requested_language=? ORDER BY l.last_collected_at DESC,t.track_id LIMIT ?",
                        (language, limit)).fetchall()
    items = []
    seen: set[str] = set()
    for row in rows:
        try:
            item = json.loads(row["raw_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        track_id = str(item.get("trackId") or "")
        if track_id and track_id not in seen:
            seen.add(track_id)
            item["_tunely_emotion"] = {
                "lyrics_availability": row["lyrics_availability"],
                "lyrics_lookup_status": row["lyrics_lookup_status"],
                "audio_analysis_status": row["audio_analysis_status"],
                "labels": json.loads(row["emotion_labels_json"] or "[]"),
                "confidence": json.loads(row["emotion_confidence_json"] or "{}"),
                "evidence": json.loads(row["emotion_evidence_json"] or "{}"),
                "model_version": row["emotion_model_version"],
                "confidence_kind": row["emotion_confidence_kind"],
            }
            items.append(item)
    return items


def dataset_statistics(conn: sqlite3.Connection) -> dict[str, Any]:
    total = conn.execute("SELECT COUNT(*) FROM itunes_tracks").fetchone()[0]
    requested = {row[0]: row[1] for row in conn.execute(
        "SELECT requested_language,COUNT(DISTINCT track_id) FROM itunes_track_languages GROUP BY requested_language")}
    verified = {row[0]: row[1] for row in conn.execute(
        "SELECT language,COUNT(*) FROM itunes_tracks WHERE language IS NOT NULL GROUP BY language")}
    category_counts = {item["id"]: 0 for item in CATEGORIES}
    for row in conn.execute("SELECT raw_json FROM itunes_tracks"):
        try:
            classification = classify_music_categories(json.loads(row[0]))
            for category in classification["category_ids"]:
                if category in category_counts:
                    category_counts[category] += 1
        except (TypeError, json.JSONDecodeError):
            continue
    fields = ("artist_name", "collection_name", "primary_genre", "release_date", "track_time_millis",
              "artwork_url", "itunes_track_url", "preview_url", "collection_id", "language")
    available = {field: conn.execute(f"SELECT COUNT(*) FROM itunes_tracks WHERE {field} IS NOT NULL").fetchone()[0]
                 for field in fields}
    lyrics_checked = conn.execute("SELECT COUNT(*) FROM itunes_tracks WHERE lyrics_lookup_status<>'not_checked'").fetchone()[0]
    lyrics_available = conn.execute("SELECT COUNT(*) FROM itunes_tracks WHERE lyrics_availability='available'").fetchone()[0]
    emotion_rows = conn.execute("SELECT emotion_labels_json FROM itunes_tracks WHERE emotion_labels_json<>'[]'").fetchall()
    emotion_counts: dict[str, int] = {}
    for row in emotion_rows:
        try:
            for label in json.loads(row[0]):
                emotion_counts[label] = emotion_counts.get(label, 0) + 1
        except (TypeError, json.JSONDecodeError):
            continue
    audio_status_counts = {row[0]: row[1] for row in conn.execute(
        "SELECT audio_analysis_status,COUNT(*) FROM itunes_tracks GROUP BY audio_analysis_status")}
    return {"track_count": total, "requested_language_counts": requested,
            "detected_language_counts": verified, "category_counts": category_counts,
            "metadata_field_counts": available,
            "emotion_analysis": {"lyrics_lookups_checked": lyrics_checked,
                                 "lyrics_available": lyrics_available,
                                 "classified_tracks": len(emotion_rows),
                                 "label_counts": emotion_counts,
                                 "audio_analysis_status_counts": audio_status_counts}}
