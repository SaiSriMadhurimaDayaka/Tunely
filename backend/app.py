"""Flask REST API for Tunely's catalog recommender and feedback learner."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from flask import Flask, jsonify, request, send_file, session
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

from backend.recommender.q_learning import (
    REWARDS,
    FeedbackConflictError,
    QLearningRecommender,
)
from backend.recommender.ranking import (
    DEFAULT_DB,
    FEATURES,
    RecommendationEngine,
    RecommendationError,
    targets_for_context,
)
from backend.recommender.saraga_import import DEFAULT_SOURCE as DEFAULT_SARAGA_SOURCE
from backend.recommender.saraga_ranking import SaragaRecommendationEngine, ensure_schema as ensure_saraga_schema
from backend.recommender.external_links import listening_links
from backend.recommender.youtube_discovery import (
    MOOD_PROFILES,
    SUPPORTED_LANGUAGES,
    GOAL_SEARCH_TERMS,
    mood_genres,
    normalize_intensity,
)
from backend.recommender.itunes_discovery import (
    ITunesDiscoveryError, ITunesSearchClient, build_itunes_query_variants,
    build_itunes_seed_query, build_itunes_term, rank_itunes_results,
)
from backend.recommender.itunes_dataset import (
    dataset_statistics, ensure_itunes_dataset_schema, local_itunes_items, upsert_itunes_tracks,
)
from backend.recommender.music_categories import CATEGORIES, infer_music_categories
from backend.recommender.emotion_pipeline import enrich_song_emotion
from backend.recommender.emotion_analysis import LocalXLMREmotionClassifier
from backend.recommender.lyrics_provider import LRCLibClient
from backend.recommender.reviewed_catalog import (
    GENRE_BY_KEY, REVIEWED_LANGUAGES, ensure_reviewed_catalog_schema, rank_reviewed_songs,
    reviewed_catalog_statistics,
)
from backend.recommender.intent_ml import load_intent_model


SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
COMMON_LANGUAGES = (
    "Bengali", "English", "Gujarati", "Hindi", "Kannada", "Malayalam",
    "Marathi", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _api_schema(database: Path) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_saraga_schema(conn)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferred_language TEXT NOT NULL DEFAULT '',
                favorite_genres_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS recommendations (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                song_id TEXT NOT NULL,
                context_json TEXT NOT NULL,
                strategy TEXT NOT NULL CHECK(strategy IN ('features', 'popularity')),
                created_at TEXT NOT NULL,
                FOREIGN KEY(song_id) REFERENCES songs(song_id)
            );
            CREATE INDEX IF NOT EXISTS idx_recommendations_user_created
                ON recommendations(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS favorites (
                favorite_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                song_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, song_id),
                FOREIGN KEY(song_id) REFERENCES songs(song_id)
            );
            CREATE INDEX IF NOT EXISTS idx_favorites_user_created
                ON favorites(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS saraga_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                context_json TEXT NOT NULL,
                strategy TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(recording_id) REFERENCES recordings(recording_id)
            );
            CREATE INDEX IF NOT EXISTS idx_saraga_recommendations_user_created
                ON saraga_recommendations(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS saraga_feedback (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                rating TEXT NOT NULL CHECK(rating IN ('loved it', 'liked it', 'it was okay', 'didn''t like it')),
                reward REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(recommendation_id) REFERENCES saraga_recommendations(recommendation_id),
                FOREIGN KEY(recording_id) REFERENCES recordings(recording_id)
            );
            CREATE INDEX IF NOT EXISTS idx_saraga_feedback_user_created
                ON saraga_feedback(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS saraga_favorites (
                favorite_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recording_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, recording_id),
                FOREIGN KEY(recording_id) REFERENCES recordings(recording_id)
            );
            CREATE INDEX IF NOT EXISTS idx_saraga_favorites_user_created
                ON saraga_favorites(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS youtube_search_cache (
                query_key TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                language TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                cached_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS youtube_song_cache (
                video_id TEXT NOT NULL,
                language TEXT NOT NULL,
                mood TEXT NOT NULL,
                intensity TEXT NOT NULL,
                title TEXT NOT NULL,
                channel_title TEXT,
                description TEXT,
                thumbnail_url TEXT,
                language_confidence REAL NOT NULL,
                language_verified INTEGER NOT NULL DEFAULT 0,
                language_evidence TEXT NOT NULL,
                genre_candidates_json TEXT NOT NULL,
                matched_genres_json TEXT NOT NULL,
                genre_confidence REAL NOT NULL,
                mood_confidence REAL NOT NULL,
                intensity_confidence REAL NOT NULL DEFAULT 0,
                source_url TEXT NOT NULL,
                youtube_music_url TEXT NOT NULL,
                metadata_source TEXT NOT NULL,
                last_verified TEXT NOT NULL,
                PRIMARY KEY(video_id, language, mood, intensity)
            );
            CREATE TABLE IF NOT EXISTS youtube_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                context_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_youtube_recommendations_user_created
                ON youtube_recommendations(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS youtube_favorites (
                favorite_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, video_id)
            );
            CREATE TABLE IF NOT EXISTS youtube_feedback (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                rating TEXT NOT NULL,
                reward REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS itunes_search_cache (
                query_key TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                language TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                cached_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS itunes_recommendations (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                context_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS itunes_discovery_state (
                user_id TEXT NOT NULL,
                context_key TEXT NOT NULL,
                next_variant INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, context_key)
            );
            CREATE INDEX IF NOT EXISTS idx_itunes_recommendations_user_created
                ON itunes_recommendations(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS itunes_favorites (
                favorite_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, track_id)
            );
            CREATE TABLE IF NOT EXISTS itunes_feedback (
                recommendation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                rating TEXT NOT NULL,
                reward REAL NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        recommendation_schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='recommendations'").fetchone()[0]
        if "exploration" not in recommendation_schema:
            conn.execute("DROP INDEX IF EXISTS idx_recommendations_user_created")
            conn.execute("ALTER TABLE recommendations RENAME TO recommendations_legacy")
            conn.execute("""CREATE TABLE recommendations (
                recommendation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, song_id TEXT NOT NULL,
                context_json TEXT NOT NULL, strategy TEXT NOT NULL CHECK(strategy IN ('features','popularity','exploration')),
                created_at TEXT NOT NULL, FOREIGN KEY(song_id) REFERENCES songs(song_id))""")
            conn.execute("""INSERT INTO recommendations (recommendation_id,user_id,song_id,context_json,strategy,created_at)
                SELECT recommendation_id,user_id,song_id,context_json,strategy,created_at FROM recommendations_legacy""")
            conn.execute("DROP TABLE recommendations_legacy")
            conn.execute("CREATE INDEX idx_recommendations_user_created ON recommendations(user_id, created_at DESC)")
        optional_columns = {"songs": ("youtube_music_url", "language", "mood", "dataset_source", "dataset_license"),
                            "recordings": ("youtube_music_url", "mood"),
                            "youtube_song_cache": ("intensity_confidence",)}
        for table, additions in optional_columns.items():
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if exists:
                columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                for column in additions:
                    if column not in columns:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")
        ensure_itunes_dataset_schema(conn)
        ensure_reviewed_catalog_schema(conn)
        conn.commit()
    finally:
        conn.close()


def _dict_rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _recording_payload(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    audio_available = item.get("audio_status") == "available_local" and bool(item.get("audio_path"))
    try:
        tempo_annotations = json.loads(item.get("tempo_annotations_json") or "[]")
    except (TypeError, json.JSONDecodeError):
        tempo_annotations = []
    title, artist = item.get("title"), item.get("artist")
    return {
        "recording_id": item.get("recording_id"), "title": item.get("title"), "artist": item.get("artist"),
        **listening_links(title, artist, item.get("youtube_music_url")),
        "mood": item.get("mood"),
        "tradition": item.get("tradition"), "release_info": item.get("release_info"),
        "raga": item.get("raga"), "tala": item.get("tala"), "laya": item.get("laya"),
        "form": item.get("form"), "work": item.get("work"), "tempo_bpm": item.get("tempo_bpm"),
        "tempo_annotations": tempo_annotations, "duration_ms": item.get("duration_ms"),
        "language": item.get("language"), "audio_available": audio_available,
        "playback_url": f"/api/audio/{item['recording_id']}" if audio_available else None,
        "audio_status": item.get("audio_status"), "dataset_source": item.get("dataset_source"),
        "dataset_license": item.get("dataset_license"), "attribution": item.get("attribution"),
    }


def create_app(database: str | Path | None = None, *, testing: bool = False) -> Flask:
    """Application factory; pass a temporary database path when testing."""
    db_path = Path(database or os.environ.get("TUNELY_DATABASE", DEFAULT_DB)).resolve()
    _api_schema(db_path)
    app = Flask(__name__)
    app.config.update(
        TESTING=testing, DATABASE=str(db_path), JSON_SORT_KEYS=False,
        SECRET_KEY=os.environ.get("TUNELY_SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("TUNELY_COOKIE_SECURE", "false").lower() == "true",
        PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30,
        ITUNES_CACHE_TTL_SECONDS=max(60, int(os.environ.get("ITUNES_CACHE_TTL_SECONDS", "900"))),
        ITUNES_RECENT_COOLDOWN_DAYS=max(1, int(os.environ.get("ITUNES_RECENT_COOLDOWN_DAYS", "14"))),
        ITUNES_SEARCH_COUNTRY=os.environ.get("ITUNES_SEARCH_COUNTRY", "in").strip().lower() or "in",
        ITUNES_SEARCH_TIMEOUT_SECONDS=max(2, min(30, float(os.environ.get("ITUNES_SEARCH_TIMEOUT_SECONDS", "8")))),
    )
    allowed_origins = [origin.strip() for origin in os.environ.get("TUNELY_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
    CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=True)
    engine = RecommendationEngine(db_path)
    saraga_engine = SaragaRecommendationEngine(db_path)
    learner = QLearningRecommender(db_path)
    app.extensions["tunely_engine"] = engine
    app.extensions["tunely_saraga_engine"] = saraga_engine
    app.extensions["tunely_learner"] = learner
    app.extensions["tunely_itunes_client"] = ITunesSearchClient(timeout=app.config["ITUNES_SEARCH_TIMEOUT_SECONDS"])
    app.extensions["tunely_lyrics_client"] = LRCLibClient(
        timeout=max(2, min(12, float(os.environ.get("TUNELY_LYRICS_TIMEOUT_SECONDS", "4"))))
    )
    app.extensions["tunely_emotion_classifier"] = LocalXLMREmotionClassifier()
    # Optional artifact: an absent/incompatible model never prevents Flask startup.
    intent_model_path = os.environ.get("TUNELY_INTENT_MODEL")
    app.extensions["tunely_intent_model"] = load_intent_model(intent_model_path) if intent_model_path else load_intent_model()

    @contextmanager
    def connect():
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def guest_session_id() -> str | None:
        candidate = request.headers.get("X-Session-ID")
        if candidate is None:
            return None
        if not SESSION_RE.fullmatch(candidate):
            raise RecommendationError("X-Session-ID may contain only letters, numbers, underscores, or hyphens (1–128 characters).")
        return candidate

    def user_id() -> str:
        account_id = session.get("account_id")
        if account_id:
            return f"account:{account_id}"
        return guest_session_id() or "anonymous"

    def authenticated_account() -> str | None:
        value = session.get("account_id")
        return str(value) if value else None

    def parse_labels(value: Any) -> list[str]:
        if value is None:
            return []
        return list(dict.fromkeys(part.strip() for part in re.split(r"[,;|]", str(value)) if part.strip()))

    def body_object() -> dict[str, Any]:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise RecommendationError("Request body must be a JSON object.")
        return body

    def error(message: str, code: str, status: int):
        return jsonify({"error": {"code": code, "message": message}}), status

    def catalog_metadata() -> dict[str, Any]:
        with connect() as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='songs'").fetchone() is None:
                return {"song_count": 0, "available_audio_features": [], "genre_available": False,
                        "release_date_available": False, "cover_image_available": False}
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(songs)")}
            song_count = int(conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0])
            reviewed_count = int(conn.execute("SELECT COUNT(*) FROM reviewed_songs").fetchone()[0])
            check_columns = [name for name in (*FEATURES, "track_genre", "release_date", "cover_image") if name in columns]
            totals = conn.execute("SELECT " + ", ".join(f"COUNT({name})" for name in check_columns) + " FROM songs").fetchone() if check_columns else []
            counts = dict(zip(check_columns, totals))
            available_features = [name for name in FEATURES if counts.get(name, 0) > 0]
            genre_count = int(conn.execute("SELECT COUNT(*) FROM songs WHERE track_genre IS NOT NULL AND TRIM(track_genre)<>''").fetchone()[0]) if "track_genre" in columns else 0
            language_count = int(conn.execute("SELECT COUNT(*) FROM songs WHERE language IS NOT NULL AND TRIM(language)<>''").fetchone()[0]) if "language" in columns else 0
            recording_count = int(conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0])
            available_traditions = [
                {"tradition": row[0], "count": int(row[1])}
                for row in conn.execute("SELECT tradition, COUNT(*) FROM recordings GROUP BY tradition ORDER BY tradition")
            ]
            known_language_count = int(conn.execute(
                "SELECT COUNT(*) FROM recordings WHERE language IS NOT NULL AND TRIM(language)<>''"
            ).fetchone()[0])
            local_audio_count = int(conn.execute(
                "SELECT COUNT(*) FROM recordings WHERE audio_status='available_local' AND audio_path IS NOT NULL"
            ).fetchone()[0])
            return {
                "song_count": reviewed_count or song_count,
                "reviewed_song_count": reviewed_count,
                "original_catalog_song_count": song_count,
                "catalog_source": "reviewed_dataset" if reviewed_count else "spotify_catalog",
                "available_audio_features": available_features,
                "audio_features_available": bool(available_features),
                "genre_available": genre_count > 0,
                "genre_count": genre_count,
                "language_count": language_count,
                "unknown_language_count": song_count - language_count,
                "release_date_available": counts.get("release_date", 0) > 0,
                "cover_image_available": counts.get("cover_image", 0) > 0,
                "saraga_recording_count": recording_count,
                "available_traditions": available_traditions,
                "verified_language_count": known_language_count,
                "saraga_audio_available_count": local_audio_count,
            }

    def feedback_summary(conn: sqlite3.Connection, owner: str, include_reward: float | None = None) -> str:
        values = [float(row[0]) for row in conn.execute(
            "SELECT reward FROM learning_feedback WHERE user_id=? ORDER BY created_at DESC LIMIT 4", (owner,)
        ).fetchall()]
        if include_reward is not None:
            values.insert(0, include_reward)
        if not values:
            return "none"
        has_positive, has_negative = any(v > 0 for v in values), any(v < 0 for v in values)
        if has_positive and has_negative:
            return "mixed"
        if has_positive:
            return "positive"
        if has_negative:
            return "negative"
        return "neutral"

    @app.errorhandler(RecommendationError)
    def handle_input_error(exc: RecommendationError):
        return error(str(exc), "invalid_request", 400)

    @app.errorhandler(FeedbackConflictError)
    def handle_feedback_conflict(exc: FeedbackConflictError):
        return error(str(exc), "duplicate_feedback", 409)

    @app.errorhandler(404)
    def handle_not_found(_exc):
        return error("The requested API route was not found.", "not_found", 404)

    @app.errorhandler(sqlite3.Error)
    def handle_database_error(exc: sqlite3.Error):
        app.logger.error("Database request failed: %s", type(exc).__name__)
        return error("The request could not be completed because of a database error.", "database_error", 503)

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        if app.testing:
            raise exc
        app.logger.exception("Unhandled API error")
        return error("The server could not complete the request.", "internal_error", 500)

    @app.get("/api/health")
    def health():
        try:
            metadata = catalog_metadata()
            return jsonify({"data": {"status": "ok", "catalog_ready": metadata["song_count"] > 0, **metadata}})
        except sqlite3.Error:
            return error("The catalog database is unavailable.", "database_error", 503)

    @app.get("/api/auth/me")
    def auth_me():
        account_id = authenticated_account()
        if not account_id:
            return jsonify({"data": {"user": None}})
        with connect() as conn:
            row = conn.execute("SELECT user_id, name, email FROM users WHERE user_id=?", (account_id,)).fetchone()
        if row is None:
            session.clear()
            return jsonify({"data": {"user": None}})
        return jsonify({"data": {"user": dict(row)}})

    def transfer_guest_data(conn: sqlite3.Connection, guest: str, account_owner: str) -> None:
        """Move this browser's existing private activity into the signed-in account."""
        if guest == account_owner:
            return
        for table, key in (("favorites", "song_id"), ("saraga_favorites", "recording_id")):
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                conn.execute(f"INSERT OR IGNORE INTO {table}(user_id,{key},created_at) SELECT ?,{key},created_at FROM {table} WHERE user_id=?", (account_owner, guest))
                conn.execute(f"DELETE FROM {table} WHERE user_id=?", (guest,))
        for table in ("recommendations", "saraga_recommendations", "saraga_feedback", "learning_feedback"):
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                conn.execute(f"UPDATE {table} SET user_id=? WHERE user_id=?", (account_owner, guest))
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='q_values'").fetchone():
            conn.execute("INSERT OR IGNORE INTO q_values(user_id,state_key,action,q_value,updated_at) SELECT ?,state_key,action,q_value,updated_at FROM q_values WHERE user_id=?", (account_owner, guest))
            conn.execute("DELETE FROM q_values WHERE user_id=?", (guest,))

    def auth_payload(body: dict[str, Any], *, registering: bool) -> tuple[str, str, str]:
        raw_name = body.get("name", "")
        raw_email = body.get("email", "")
        if registering and not isinstance(raw_name, str):
            raise RecommendationError("Name must be text.")
        if not isinstance(raw_email, str):
            raise RecommendationError("Email must be text.")
        name = raw_name.strip() if isinstance(raw_name, str) else ""
        email = raw_email.strip().casefold()
        password = body.get("password")
        if registering and (not name or len(name) > 100):
            raise RecommendationError("Name must contain between 1 and 100 characters.")
        if len(email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            raise RecommendationError("Enter a valid email address.")
        if not isinstance(password, str) or len(password) < 8 or len(password) > 128:
            raise RecommendationError("Password must be between 8 and 128 characters.")
        return name, email, password

    @app.post("/api/auth/register")
    def auth_register():
        if authenticated_account():
            return error("Sign out before creating or signing into a different account.", "already_authenticated", 409)
        body = body_object()
        name, email, password = auth_payload(body, registering=True)
        account_id = str(uuid.uuid4())
        try:
            with connect() as conn:
                conn.execute("INSERT INTO users(user_id,name,email,password_hash,created_at) VALUES (?,?,?,?,?)",
                             (account_id, name, email, generate_password_hash(password), _utc_now()))
                conn.execute("INSERT INTO user_preferences(user_id,updated_at) VALUES (?,?)", (account_id, _utc_now()))
                guest = guest_session_id()
                if guest:
                    transfer_guest_data(conn, guest, f"account:{account_id}")
        except sqlite3.IntegrityError:
            return error("An account with this email already exists. Try signing in.", "email_in_use", 409)
        session.clear()
        session["account_id"] = account_id
        session.permanent = True
        return jsonify({"data": {"user": {"user_id": account_id, "name": name, "email": email}}}), 201

    @app.post("/api/auth/login")
    def auth_login():
        if authenticated_account():
            return error("Sign out before creating or signing into a different account.", "already_authenticated", 409)
        _, email, password = auth_payload(body_object(), registering=False)
        with connect() as conn:
            row = conn.execute("SELECT user_id,name,email,password_hash FROM users WHERE email=?", (email,)).fetchone()
            if row is None or not check_password_hash(row["password_hash"], password):
                return error("Email or password is incorrect.", "invalid_credentials", 401)
            guest = guest_session_id()
            if guest:
                transfer_guest_data(conn, guest, f"account:{row['user_id']}")
            user = {"user_id": row["user_id"], "name": row["name"], "email": row["email"]}
        session.clear()
        session["account_id"] = user["user_id"]
        session.permanent = True
        return jsonify({"data": {"user": user}})

    @app.post("/api/auth/logout")
    def auth_logout():
        session.clear()
        return jsonify({"data": {"logged_out": True}})

    @app.get("/api/preferences")
    def get_preferences():
        account_id = authenticated_account()
        if not account_id:
            return error("Sign in to access saved preferences.", "authentication_required", 401)
        with connect() as conn:
            row = conn.execute("SELECT preferred_language,favorite_genres_json FROM user_preferences WHERE user_id=?", (account_id,)).fetchone()
        return jsonify({"data": {"preferred_language": row["preferred_language"], "favorite_genres": json.loads(row["favorite_genres_json"]) } if row else {"preferred_language": "", "favorite_genres": []}})

    @app.put("/api/preferences")
    def put_preferences():
        account_id = authenticated_account()
        if not account_id:
            return error("Sign in to save preferences.", "authentication_required", 401)
        body = body_object()
        language = body.get("preferred_language", "")
        genres = body.get("favorite_genres", [])
        if not isinstance(language, str) or len(language) > 100:
            raise RecommendationError("preferred_language must be a string no longer than 100 characters.")
        if not isinstance(genres, list) or len(genres) > 50 or any(not isinstance(item, str) or len(item) > 80 for item in genres):
            raise RecommendationError("favorite_genres must be a list of at most 50 genre names.")
        with connect() as conn:
            conn.execute("INSERT INTO user_preferences(user_id,preferred_language,favorite_genres_json,updated_at) VALUES (?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET preferred_language=excluded.preferred_language,favorite_genres_json=excluded.favorite_genres_json,updated_at=excluded.updated_at",
                         (account_id, language.strip(), json.dumps(list(dict.fromkeys(item.strip() for item in genres if item.strip()))), _utc_now()))
        return jsonify({"data": {"preferred_language": language.strip(), "favorite_genres": genres}})

    @app.get("/api/songs")
    def songs():
        try:
            limit = int(request.args.get("limit", "20"))
            offset = int(request.args.get("offset", "0"))
        except ValueError as exc:
            raise RecommendationError("limit and offset must be integers.") from exc
        if not 1 <= limit <= 100 or offset < 0:
            raise RecommendationError("limit must be 1–100 and offset must be non-negative.")
        genre = request.args.get("genre", "").strip()
        language = request.args.get("language", "").strip()
        query = request.args.get("q", "").strip()
        where: list[str] = []
        params: list[Any] = []
        if genre:
            if not catalog_metadata()["genre_available"]:
                raise RecommendationError("Genre filtering is unavailable: the selected catalog has no genre values.")
            where.append("LOWER(track_genre)=LOWER(?)")
            params.append(genre)
        if language and language.casefold() == "unknown":
            where.append("(language IS NULL OR TRIM(language)='')")
        elif language and language.casefold() != "all languages":
            where.append("(',' || replace(replace(lower(COALESCE(language,'')), ';', ','), '|', ',') || ',') LIKE ?")
            params.append(f"%,{language.casefold()},%")
        if query:
            where.append("(song_name LIKE ? ESCAPE '\\' OR artist LIKE ? ESCAPE '\\')")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.extend((f"%{escaped}%", f"%{escaped}%"))
        clause = " WHERE " + " AND ".join(where) if where else ""
        with connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM songs{clause}", params).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM songs{clause} ORDER BY song_name COLLATE NOCASE, artist COLLATE NOCASE LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return jsonify({"data": _dict_rows(rows), "meta": {"limit": limit, "offset": offset, "total": total}})

    @app.get("/api/traditions")
    def traditions():
        with connect() as conn:
            rows = conn.execute(
                "SELECT tradition, COUNT(*) AS recording_count FROM recordings GROUP BY tradition ORDER BY tradition"
            ).fetchall()
        return jsonify({"data": _dict_rows(rows)})

    @app.get("/api/languages")
    def languages():
        with connect() as conn:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(songs)")}
            labels = []
            if "language" in columns:
                labels = [row[0] for row in conn.execute("SELECT language FROM songs WHERE language IS NOT NULL AND TRIM(language)<>''")]
            counts: dict[str, int] = {}
            for value in labels:
                for label in parse_labels(value):
                    counts[label] = counts.get(label, 0) + 1
            has_verified_languages = bool(counts)
            unknown = int(conn.execute("SELECT COUNT(*) FROM songs WHERE language IS NULL OR TRIM(language)='' ").fetchone()[0]) if "language" in columns else int(conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0])
        # Keep useful language filters visible even when this imported catalog has
        # no language annotations. A zero count is explicit; it never assigns a
        # language to a song.
        for language in COMMON_LANGUAGES:
            counts.setdefault(language, 0)
        data = [{"language": language, "song_count": count} for language, count in sorted(counts.items(), key=lambda item: item[0].casefold())]
        if unknown:
            data.append({"language": "Unknown", "song_count": unknown})
        return jsonify({"data": data, "meta": {"available": has_verified_languages, "unknown_count": unknown}})

    @app.get("/api/discovery/languages")
    def discovery_languages():
        """Expose reviewed languages when imported, otherwise preserve the iTunes discovery list."""
        with connect() as conn:
            reviewed_count = int(conn.execute("SELECT COUNT(*) FROM reviewed_songs").fetchone()[0])
            counts = {row[0]: row[1] for row in conn.execute(
                "SELECT language,COUNT(*) FROM reviewed_songs GROUP BY language")}
        if reviewed_count:
            languages = [
                {"language": name, "language_code": SUPPORTED_LANGUAGES[name]["code"],
                 "search_supported": True, "catalog_song_count": counts.get(name, 0)}
                for name in REVIEWED_LANGUAGES
            ]
            source = "manually reviewed Tunely dataset; language values come from its Language column"
        else:
            languages = [
                {"language": name, "language_code": values["code"], "search_supported": True}
                for name, values in SUPPORTED_LANGUAGES.items()
            ]
            source = "iTunes Search API; language is accepted only when catalog text provides evidence"
        return jsonify({"data": [
            item for item in languages
        ], "meta": {"source": source}})

    @app.get("/api/reviewed-dataset/stats")
    def reviewed_dataset_stats():
        with connect() as conn:
            return jsonify({"data": reviewed_catalog_statistics(conn), "meta": {
                "source": "manually reviewed workbook", "genre_source_of_truth": "Primary Genre",
                "old_cutoff": "release year 2009 or earlier"}})

    @app.get("/api/discovery/categories")
    def discovery_categories():
        """Evidence-based multi-label categories available in catalog results."""
        return jsonify({"data": list(CATEGORIES), "meta": {
            "method": "release dates, provider genre metadata, and optional supplied audio or lyric evidence",
            "limitations": "iTunes Search does not provide lyrics or audio features; unsupported labels are omitted.",
        }})

    @app.get("/api/itunes-dataset/stats")
    def itunes_dataset_stats():
        with connect() as conn:
            return jsonify({"data": dataset_statistics(conn), "meta": {
                "source": "local iTunes Search API metadata",
                "music_categories": "evidence-based multi-label rules using provider metadata, release dates, and optional supplied audio or lyric cues",
                "emotion_detection": "LRCLIB match plus multilingual lyric lexicon; an existing local fine-tuned XLM-R checkpoint is optional, and lexicon support scores are not calibrated probabilities",
                "limitations": "No XLM-R weights are downloaded, no large dataset is collected for this feature, lyrics text is not persisted, and iTunes preview audio is not fetched or analyzed.",
            }})

    @app.post("/api/mood-recommendations")
    def mood_recommendations():
        """Recommend reviewed tracks first; retain iTunes discovery when no reviewed catalog is loaded."""
        body = body_object()
        language = body.get("language")
        mood = str(body.get("mood") or "").strip().casefold()
        requested_genre = body.get("genre", body.get("category"))
        intensity_level, intensity = normalize_intensity(body.get("intensity"))
        if not isinstance(language, str) or language.strip().casefold() not in {name.casefold() for name in SUPPORTED_LANGUAGES}:
            raise RecommendationError(f"Choose a supported language: {', '.join(SUPPORTED_LANGUAGES)}.")
        language = next(name for name in SUPPORTED_LANGUAGES if name.casefold() == language.strip().casefold())
        if requested_genre is None and mood not in MOOD_PROFILES and mood in GENRE_BY_KEY:
            requested_genre = GENRE_BY_KEY[mood]
            mood = "neutral"
        elif requested_genre is not None:
            if not isinstance(requested_genre, str) or requested_genre.strip().casefold() not in GENRE_BY_KEY:
                raise RecommendationError(f"genre must be one of: {', '.join(GENRE_BY_KEY.values())}.")
            requested_genre = GENRE_BY_KEY[requested_genre.strip().casefold()]
        if mood not in MOOD_PROFILES:
            raise RecommendationError(f"Choose a supported mood: {', '.join(MOOD_PROFILES)}.")
        if not intensity_level:
            raise RecommendationError("intensity must be Mild, Moderate, or Intense.")
        goal = str(body.get("goal") or "").strip().casefold()
        if goal and goal not in GOAL_SEARCH_TERMS:
            raise RecommendationError(f"goal must be one of: {', '.join(GOAL_SEARCH_TERMS)}.")
        preference = body.get("preference", "")
        if not isinstance(preference, str):
            raise RecommendationError("preference must be text.")
        preference = " ".join(preference.split())[:500]
        limit = body.get("limit", 8)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise RecommendationError("limit must be an integer between 1 and 20.")

        with connect() as conn:
            reviewed_count = int(conn.execute("SELECT COUNT(*) FROM reviewed_songs").fetchone()[0])
        if requested_genre and not reviewed_count:
            raise RecommendationError("Reviewed genre filtering is unavailable because no reviewed catalog is imported.")
        if reviewed_count:
            if requested_genre and requested_genre not in GENRE_BY_KEY.values():
                raise RecommendationError(f"genre must be one of: {', '.join(GENRE_BY_KEY.values())}.")
            supported = {name.casefold(): name for name in REVIEWED_LANGUAGES}
            if language.casefold() not in supported:
                raise RecommendationError(f"Choose one of the reviewed catalog languages: {', '.join(REVIEWED_LANGUAGES)}.")
            language = supported[language.casefold()]
            supplied_exclusions = body.get("exclude_track_ids", [])
            if not isinstance(supplied_exclusions, list):
                supplied_exclusions = []
            exclusions = list(dict.fromkeys(str(value) for value in supplied_exclusions if str(value)))[:100]
            result_context = {"language": language, "mood": mood, "intensity": intensity,
                              "intensity_level": intensity_level, "goal": goal or None,
                              "preference": preference or None,
                              "selected_genre": requested_genre,
                              "catalog_source": "reviewed_dataset"}
            model = app.extensions.get("tunely_intent_model")
            model_scores = None
            model_used = False
            if model is not None and requested_genre is None:
                try:
                    model_preference = preference or GOAL_SEARCH_TERMS.get(goal, "")
                    model_scores = model.predict_scores(model_preference, mood, intensity)
                    model_used = True
                except Exception:
                    # The long-standing deterministic genre targeting remains the fallback.
                    app.logger.exception("Intent genre model inference failed; using existing ranking targets")
            with connect() as conn:
                ranked = rank_reviewed_songs(conn, user_id=user_id(), language=language, mood=mood,
                    intensity=intensity, goal=goal, limit=limit, exclude_track_ids=exclusions,
                    recent_days=app.config["ITUNES_RECENT_COOLDOWN_DAYS"], genre_filter=requested_genre,
                    genre_targets_override=model_scores)
                saved = []
                timestamp = _utc_now()
                for song in ranked["recommendations"]:
                    recommendation_id = str(uuid.uuid4())
                    song["recommendation_id"] = recommendation_id
                    if song.get("itunes_track_id"):
                        conn.execute("""INSERT INTO itunes_recommendations
                            (recommendation_id,user_id,track_id,context_json,result_json,score,created_at)
                            VALUES(?,?,?,?,?,?,?)""", (recommendation_id, user_id(), song["itunes_track_id"],
                            json.dumps(result_context, ensure_ascii=False), json.dumps(song, ensure_ascii=False),
                            song["score"], timestamp))
                    saved.append(song)
            if saved:
                detail = f" with the {requested_genre} genre" if requested_genre else ""
                message = f"Found {len(saved)} matches from your manually reviewed {language} catalog{detail}."
                if len(saved) < limit:
                    message += f" Only {len(saved)} eligible tracks remain for this selection."
            elif ranked["language_catalog_count"] == 0:
                message = f"Your reviewed catalog has no songs labeled {language}. No songs from another language were substituted."
            elif requested_genre:
                message = f"Your reviewed {language} catalog has no songs tagged {requested_genre}."
            else:
                message = f"No reviewed {language} songs match the {mood} mood and listening goal in this catalog. Try another goal or refresh."
            return jsonify({"data": {"recommendations": saved, "language": language,
                "mood": mood, "genre": requested_genre, "intensity": intensity, "goal": goal or None,
                "inferred_categories": list(ranked["genre_targets"]),
                "inferred_category_labels": list(ranked["genre_targets"]),
                "genre_relevance_scores": ranked.get("genre_relevance_scores", ranked["genre_targets"]),
                "recommendation_model": "tfidf_ovr_logistic_regression" if model_used else "existing_rule_based_fallback",
                "model_version": model.bundle.get("model_version") if model_used else None,
                "api_source": "reviewed_dataset", "discovery_source": "reviewed_dataset",
                "local_dataset_count": ranked["language_catalog_count"],
                "candidate_count": ranked["candidate_count"], "rotation_fallback": ranked["rotation_fallback"],
                "message": message}})

        inferred_categories = infer_music_categories(mood, goal)
        # The broad language query protects recall. A rotating mood/goal/category
        # query and a provider-derived artist/album query expand discovery.
        query, genres = build_itunes_term(language, mood, intensity, goal)
        variants = build_itunes_query_variants(language, mood, intensity, goal, inferred_categories)
        owner = user_id()
        context_key = hashlib.sha256(json.dumps(
            [language.casefold(), mood, intensity.casefold(), goal], separators=(",", ":")
        ).encode("utf-8")).hexdigest()
        with connect() as conn:
            state = conn.execute("SELECT next_variant FROM itunes_discovery_state WHERE user_id=? AND context_key=?",
                                 (owner, context_key)).fetchone()
            rotation = int(state["next_variant"]) if state else 0
            conn.execute("INSERT INTO itunes_discovery_state(user_id,context_key,next_variant,updated_at) VALUES(?,?,?,?) "
                         "ON CONFLICT(user_id,context_key) DO UPDATE SET next_variant=excluded.next_variant,updated_at=excluded.updated_at",
                         (owner, context_key, rotation + 1, _utc_now()))
        variant = variants[rotation % len(variants)] if variants else query
        force_refresh = bool(body.get("refresh"))
        requested_exclusions = body.get("exclude_track_ids", [])
        if not isinstance(requested_exclusions, list):
            requested_exclusions = []
        requested_exclusions = list(dict.fromkeys(str(value) for value in requested_exclusions
                                                  if str(value).isdigit()))[:100]
        cutoff = (datetime.now(timezone.utc) - timedelta(days=app.config["ITUNES_RECENT_COOLDOWN_DAYS"])).isoformat(timespec="seconds")
        recent_track_ids: list[str] = []
        recent_artists: list[str] = []
        recent_albums: list[str] = []
        with connect() as conn:
            recent_rows = conn.execute("SELECT track_id,result_json FROM itunes_recommendations "
                                       "WHERE user_id=? AND created_at>=? ORDER BY created_at DESC LIMIT 500",
                                       (owner, cutoff)).fetchall()
            local_items = local_itunes_items(conn, language, limit=500)
        for row in recent_rows:
            recent_track_ids.append(str(row["track_id"]))
            try:
                recent_song = json.loads(row["result_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if recent_song.get("artist"):
                recent_artists.append(str(recent_song["artist"]))
            if recent_song.get("album_name"):
                recent_albums.append(str(recent_song["album_name"]))
        client = app.extensions["tunely_itunes_client"]
        now = time.time()
        should_discover = force_refresh or len(local_items) < 60
        queries = list(dict.fromkeys((query, variant))) if should_discover else []
        items: list[dict[str, Any]] = list(local_items)
        cached_queries = 0
        successful_queries = 0
        discovery_error: ITunesDiscoveryError | None = None
        try:
            for search_query in queries:
                query_key = hashlib.sha256(f"{search_query.casefold()}|{app.config['ITUNES_SEARCH_COUNTRY']}".encode("utf-8")).hexdigest()
                with connect() as conn:
                    cached = conn.execute("SELECT payload_json,expires_at FROM itunes_search_cache WHERE query_key=?", (query_key,)).fetchone()
                query_items = None
                if not force_refresh and cached and float(cached["expires_at"]) > now:
                    try:
                        cached_items = json.loads(cached["payload_json"])
                        if isinstance(cached_items, list) and all(isinstance(item, dict) for item in cached_items):
                            query_items = cached_items
                    except json.JSONDecodeError:
                        logging.warning("Discarding unreadable iTunes cache for query %r", search_query)
                if query_items is not None:
                    cached_queries += 1
                    successful_queries += 1
                else:
                    try:
                        query_items = client.search(search_query, app.config["ITUNES_SEARCH_COUNTRY"], 50)
                        successful_queries += 1
                        encoded = json.dumps(query_items, ensure_ascii=False)
                        with connect() as conn:
                            conn.execute("INSERT INTO itunes_search_cache(query_key,query_text,language,payload_json,cached_at,expires_at) VALUES(?,?,?,?,?,?) ON CONFLICT(query_key) DO UPDATE SET query_text=excluded.query_text,language=excluded.language,payload_json=excluded.payload_json,cached_at=excluded.cached_at,expires_at=excluded.expires_at",
                                         (query_key, search_query, language, encoded, now, now + app.config["ITUNES_CACHE_TTL_SECONDS"]))
                            upsert_itunes_tracks(conn, query_items, requested_language=language,
                                                 country=app.config["ITUNES_SEARCH_COUNTRY"])
                    except ITunesDiscoveryError as exc:
                        logging.warning("iTunes query failed (%s) for %r", exc.code, search_query)
                        discovery_error = exc
                        continue
                items.extend(query_items)
                # Results from broad/current searches seed one metadata-derived query.
                seed_query = build_itunes_seed_query(language, items, recent_artists=recent_artists,
                                                     recent_albums=recent_albums, rotation=rotation)
                if seed_query and seed_query not in queries and len(queries) < 3:
                    queries.append(seed_query)
            if successful_queries == 0 and discovery_error and not local_items:
                raise discovery_error
            with connect() as conn:
                conn.execute("DELETE FROM itunes_search_cache WHERE expires_at<?", (now,))
            recommendations = rank_itunes_results(items, language=language, mood=mood,
                                                  intensity=intensity, genres=genres,
                                                  limit=limit, goal=goal,
                                                  preferred_categories=inferred_categories,
                                                  recent_track_ids=recent_track_ids,
                                                  exclude_track_ids=requested_exclusions,
                                                  recent_artists=recent_artists,
                                                  recent_albums=recent_albums)
        except ITunesDiscoveryError as exc:
            return jsonify({"data": {"recommendations": [], "language": language, "mood": mood,
                "intensity": intensity, "genre_candidates": genres, "api_source": "itunes",
                "inferred_categories": inferred_categories,
                "error_code": exc.code, "message": str(exc)}})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logging.exception("Could not read the iTunes discovery cache")
            return jsonify({"data": {"recommendations": [], "language": language, "mood": mood,
                "intensity": intensity, "genre_candidates": genres, "api_source": "itunes",
                "error_code": "catalog_error", "message": "Tunely could not read music search results. Please try again."}})

        # Lookup is sequential and cached per normalized title/artist/album. Only
        # public lyrics metadata and classifier output are persisted; lyric text
        # stays in memory for this classification call.
        with connect() as conn:
            for index, song in enumerate(recommendations):
                if index and not app.testing:
                    time.sleep(0.25)  # LRCLIB requests a short delay for batches.
                enrich_song_emotion(
                    conn, song, language,
                    provider=app.extensions["tunely_lyrics_client"],
                    classifier=app.extensions["tunely_emotion_classifier"],
                )

        if not recommendations:
            if mood == "nostalgic" and goal != "discover something new":
                message = "No recommendations matched this nostalgic request. Tracks labeled Latest are excluded here; try Refresh or another language."
            else:
                message = (f"The iTunes catalog returned no songs for {language} with this mood and listening goal. Try a different selection."
                           if not items else "The catalog returned entries without usable song titles or track IDs. Please retry the search.")
        else:
            message = f"Found {len(recommendations)} catalog matches. Language labels are estimates from visible text, not verified track language metadata."
            if local_items and discovery_error:
                message += " iTunes was unavailable, so these matches came from Tunely's saved catalog."
        context = {"language": language, "mood": mood, "intensity": intensity,
                   "intensity_level": intensity_level, "goal": goal or None,
                   "inferred_categories": inferred_categories,
                   "genre_candidates": genres, "search_queries": queries,
                   "discovery_rotation": rotation, "refresh": force_refresh}
        timestamp = _utc_now()
        saved = []
        with connect() as conn:
            for song in recommendations:
                recommendation_id = str(uuid.uuid4())
                song["recommendation_id"] = recommendation_id
                conn.execute("INSERT INTO itunes_recommendations(recommendation_id,user_id,track_id,context_json,result_json,score,created_at) VALUES(?,?,?,?,?,?,?)",
                             (recommendation_id, owner, song["itunes_track_id"],
                              json.dumps(context, ensure_ascii=False), json.dumps(song, ensure_ascii=False),
                              song["score"], timestamp))
                saved.append(song)
        return jsonify({"data": {"recommendations": saved, "language": language,
            "mood": mood, "intensity": intensity, "genre_candidates": genres,
            "inferred_categories": inferred_categories,
            "inferred_category_labels": [next(item["label"] for item in CATEGORIES if item["id"] == category)
                                          for category in inferred_categories],
            "api_source": "itunes", "cache_hit": cached_queries == len(queries),
            "cached_query_count": cached_queries, "search_queries": queries,
            "search_query": query, "discovery_rotation": rotation,
            "discovery_source": "local_dataset_and_itunes_api" if should_discover else "local_dataset",
            "local_dataset_count": len(local_items),
            "message": message}})

    @app.get("/api/genres")
    def genres():
        with connect() as conn:
            rows = conn.execute("SELECT track_genre,COUNT(*) AS song_count FROM songs WHERE track_genre IS NOT NULL AND TRIM(track_genre)<>'' GROUP BY track_genre ORDER BY track_genre COLLATE NOCASE").fetchall()
        return jsonify({"data": _dict_rows(rows), "meta": {"available": bool(rows)}})

    @app.get("/api/recordings")
    def recordings():
        try:
            limit = int(request.args.get("limit", "20"))
            offset = int(request.args.get("offset", "0"))
        except ValueError as exc:
            raise RecommendationError("limit and offset must be integers.") from exc
        if not 1 <= limit <= 100 or offset < 0:
            raise RecommendationError("limit must be 1–100 and offset must be non-negative.")
        clauses: list[str] = []
        values: list[Any] = []
        for parameter, column in (("tradition", "tradition"), ("language", "language")):
            value = request.args.get(parameter, "").strip()
            if value:
                if parameter == "language":
                    with connect() as conn:
                        has_language = conn.execute(
                            "SELECT 1 FROM recordings WHERE language IS NOT NULL AND TRIM(language)<>'' LIMIT 1"
                        ).fetchone()
                    if has_language is None:
                        raise RecommendationError("Language filtering is unavailable: Saraga has no verified language values.")
                clauses.append(f"LOWER({column})=LOWER(?)")
                values.append(value)
        for parameter, column in (("raga", "raga"), ("tala", "tala")):
            value = request.args.get(parameter, "").strip()
            if value:
                clauses.append(f"LOWER({column}) LIKE ? ESCAPE '\\'")
                escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                values.append(f"%{escaped.lower()}%")
        query = request.args.get("q", "").strip()
        if query:
            clauses.append("(LOWER(title) LIKE ? ESCAPE '\\' OR LOWER(COALESCE(artist,'')) LIKE ? ESCAPE '\\')")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_").lower()
            values.extend((f"%{escaped}%", f"%{escaped}%"))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(*) FROM recordings{where}", values).fetchone()[0])
            rows = conn.execute(
                f"SELECT * FROM recordings{where} ORDER BY title COLLATE NOCASE, recording_id LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()
        return jsonify({"data": [_recording_payload(row) for row in rows], "meta": {"limit": limit, "offset": offset, "total": total}})

    @app.get("/api/recordings/<recording_id>")
    def recording_detail(recording_id: str):
        with connect() as conn:
            row = conn.execute("SELECT * FROM recordings WHERE recording_id=?", (recording_id,)).fetchone()
        if row is None:
            return error("Recording was not found in the Saraga catalog.", "recording_not_found", 404)
        return jsonify({"data": _recording_payload(row)})

    @app.post("/api/saraga/recommendations")
    def saraga_recommendations():
        body = body_object()
        owner = user_id()
        raw_context = body.get("context", body)
        if not isinstance(raw_context, dict):
            raise RecommendationError("context must be a JSON object.")
        context = {key: raw_context[key] for key in ("mood", "mood_level", "goal", "tradition", "language", "preferences") if key in raw_context}
        limit = body.get("limit", 8)
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise RecommendationError("limit must be an integer between 1 and 100.")
        with connect() as conn:
            previous_ids = {str(row[0]) for row in conn.execute(
                "SELECT DISTINCT recording_id FROM saraga_recommendations WHERE user_id=?", (owner,)
            )}
        result = saraga_engine.recommend(context, limit=limit, exclude_ids=previous_ids)
        result["rotation_reset"] = False
        if not result["recommendations"] and previous_ids:
            result = saraga_engine.recommend(context, limit=limit)
            result["rotation_reset"] = bool(result["recommendations"])
            if result["rotation_reset"]:
                result["message"] = "You have seen all matching recordings in this session. Starting a fresh rotation."
        now = _utc_now()
        saved: list[dict[str, Any]] = []
        with connect() as conn:
            for recording in result["recommendations"]:
                recommendation_id = str(uuid.uuid4())
                conn.execute("""INSERT INTO saraga_recommendations
                    (recommendation_id, user_id, recording_id, context_json, strategy, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (recommendation_id, owner, recording["recording_id"], json.dumps(context, ensure_ascii=False), result["strategy"], now))
                recording["recommendation_id"] = recommendation_id
                saved.append(recording)
        result["recommendations"] = saved
        return jsonify({"data": result})

    @app.get("/api/audio/<recording_id>")
    def play_recording(recording_id: str):
        with connect() as conn:
            row = conn.execute(
                "SELECT audio_path, audio_status FROM recordings WHERE recording_id=?", (recording_id,)
            ).fetchone()
        if row is None or row["audio_status"] != "available_local" or not row["audio_path"]:
            return error("Audio is not available locally for this recording.", "audio_unavailable", 404)
        audio_root = Path(os.environ.get("TUNELY_SARAGA_SOURCE", str(DEFAULT_SARAGA_SOURCE))).resolve()
        audio_file = (audio_root / row["audio_path"]).resolve()
        try:
            audio_file.relative_to(audio_root)
        except ValueError:
            return error("Audio is not available for this recording.", "audio_unavailable", 404)
        if not audio_file.is_file():
            return error("The local audio file is missing.", "audio_unavailable", 404)
        return send_file(audio_file, mimetype="audio/mpeg", conditional=True, as_attachment=False)

    @app.get("/api/recording-favorites")
    def get_recording_favorites():
        owner = user_id()
        with connect() as conn:
            rows = conn.execute("""SELECT r.*, f.created_at AS favorited_at
                FROM saraga_favorites f JOIN recordings r ON r.recording_id=f.recording_id
                WHERE f.user_id=? ORDER BY f.created_at DESC""", (owner,)).fetchall()
        return jsonify({"data": [_recording_payload(row) | {"favorited_at": row["favorited_at"]} for row in rows]})

    @app.post("/api/recording-favorites")
    def add_recording_favorite():
        owner = user_id()
        body = body_object()
        recording_id = body.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id.strip():
            raise RecommendationError("recording_id is required.")
        with connect() as conn:
            if conn.execute("SELECT 1 FROM recordings WHERE recording_id=?", (recording_id,)).fetchone() is None:
                return error("Recording was not found in the Saraga catalog.", "recording_not_found", 404)
            cursor = conn.execute("INSERT OR IGNORE INTO saraga_favorites(user_id, recording_id, created_at) VALUES (?, ?, ?)",
                                  (owner, recording_id, _utc_now()))
        return jsonify({"data": {"recording_id": recording_id, "saved": True, "already_saved": cursor.rowcount == 0}}), 201

    @app.delete("/api/recording-favorites/<recording_id>")
    def remove_recording_favorite(recording_id: str):
        owner = user_id()
        with connect() as conn:
            cursor = conn.execute("DELETE FROM saraga_favorites WHERE user_id=? AND recording_id=?", (owner, recording_id))
        if cursor.rowcount == 0:
            return error("Saved recording was not found for this session.", "favorite_not_found", 404)
        return jsonify({"data": {"recording_id": recording_id, "saved": False}})

    @app.post("/api/recording-feedback")
    def recording_feedback():
        owner = user_id()
        body = body_object()
        recommendation_id = body.get("recommendation_id")
        rating = body.get("rating")
        if not isinstance(recommendation_id, str) or not recommendation_id:
            raise RecommendationError("recommendation_id is required.")
        if not isinstance(rating, str) or rating.strip().lower() not in REWARDS:
            raise RecommendationError(f"rating must be one of: {', '.join(REWARDS)}.")
        normalized = rating.strip().lower()
        with connect() as conn:
            row = conn.execute("SELECT recording_id FROM saraga_recommendations WHERE recommendation_id=? AND user_id=?",
                               (recommendation_id, owner)).fetchone()
            if row is None:
                return error("Recommendation was not found for this session.", "recommendation_not_found", 404)
            try:
                conn.execute("""INSERT INTO saraga_feedback
                    (recommendation_id, user_id, recording_id, rating, reward, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (recommendation_id, owner, row["recording_id"], normalized, REWARDS[normalized], _utc_now()))
            except sqlite3.IntegrityError:
                return error("Feedback was already submitted for this recommendation.", "duplicate_feedback", 409)
        return jsonify({"data": {"recommendation_id": recommendation_id, "recording_id": row["recording_id"],
                                 "rating": normalized, "reward": REWARDS[normalized],
                                 "learning_status": "feedback_saved; Saraga policy learning not enabled yet"}}), 201

    @app.get("/api/recording-history")
    def recording_history():
        owner = user_id()
        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError as exc:
            raise RecommendationError("limit must be an integer.") from exc
        if not 1 <= limit <= 100:
            raise RecommendationError("limit must be between 1 and 100.")
        with connect() as conn:
            rows = conn.execute("""SELECT r.*, rr.recommendation_id, rr.context_json, rr.strategy, rr.created_at AS recommended_at,
                    f.rating, f.reward
                FROM saraga_recommendations rr JOIN recordings r ON r.recording_id=rr.recording_id
                LEFT JOIN saraga_feedback f ON f.recommendation_id=rr.recommendation_id
                WHERE rr.user_id=? ORDER BY rr.created_at DESC LIMIT ?""", (owner, limit)).fetchall()
        items = []
        for row in rows:
            item = _recording_payload(row)
            item.update({"recommendation_id": row["recommendation_id"], "context": json.loads(row["context_json"]),
                         "strategy": row["strategy"], "recommended_at": row["recommended_at"],
                         "rating": row["rating"], "reward": row["reward"]})
            items.append(item)
        return jsonify({"data": items, "meta": {"limit": limit}})

    @app.get("/api/recording-insights")
    def recording_insights():
        """Summarize Saraga interactions without implying a trained policy."""
        owner = user_id()
        with connect() as conn:
            recommendation_rows = conn.execute(
                "SELECT context_json FROM saraga_recommendations WHERE user_id=? ORDER BY created_at DESC",
                (owner,),
            ).fetchall()
            feedback_count = int(conn.execute(
                "SELECT COUNT(*) FROM saraga_feedback WHERE user_id=?", (owner,)
            ).fetchone()[0])
            positive_count = int(conn.execute(
                "SELECT COUNT(*) FROM saraga_feedback WHERE user_id=? AND reward>0", (owner,)
            ).fetchone()[0])
        goal_counts: dict[str, int] = {}
        mood_counts: dict[tuple[str, int | None], int] = {}
        for row in recommendation_rows:
            try:
                context = json.loads(row["context_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            goal = str(context.get("goal") or "Unspecified goal")
            goal_counts[goal] = goal_counts.get(goal, 0) + 1
            mood = str(context.get("mood") or "Unspecified")
            level = context.get("mood_level")
            if not isinstance(level, int) or isinstance(level, bool):
                level = None
            key = (mood, level)
            mood_counts[key] = mood_counts.get(key, 0) + 1
        return jsonify({"data": {
            "selected_goals": [{"goal": goal, "recommendations": count}
                               for goal, count in sorted(goal_counts.items(), key=lambda item: (-item[1], item[0]))],
            "mood_levels": [{"mood": mood, "level": level, "selections": count}
                            for (mood, level), count in sorted(mood_counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1] or 0))],
            "feedback_and_q_learning": {"feedback_count": feedback_count, "positive_feedback_count": positive_count,
                                        "q_values": [], "policy_status": "not_enabled"},
            "positive_feedback_characteristics": {"rated_songs": 0, "average_audio_features_on_positive_feedback": {}},
            "reliability_note": "These are descriptive counts from this session. Saraga ratings are stored, but they do not train a recommendation policy.",
        }})

    @app.post("/api/recommendations")
    def recommendations():
        body = body_object()
        owner = user_id()
        raw_context = body.get("context", body)
        if not isinstance(raw_context, dict):
            raise RecommendationError("context must be a JSON object.")
        context = {key: raw_context[key] for key in ("mood", "mood_level", "goal", "genre", "language", "explicit", "feature_preferences") if key in raw_context}
        try:
            limit = body.get("limit", 12)
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise RecommendationError("limit must be an integer between 1 and 100.")
            feature_preferences = context.get("feature_preferences") or {}
            if not isinstance(feature_preferences, dict):
                raise RecommendationError("feature_preferences must be an object.")
            feature_preferences = {str(key).strip().lower(): value for key, value in feature_preferences.items()}
            if "feature_preferences" in context:
                context["feature_preferences"] = feature_preferences
            with connect() as conn:
                context["feedback_summary"] = feedback_summary(conn, owner)
                metadata = catalog_metadata()
                previously_recommended = {
                    str(row[0]) for row in conn.execute(
                        "SELECT DISTINCT song_id FROM recommendations WHERE user_id=?", (owner,)
                    )
                }
            available_features = set(metadata["available_audio_features"])
            base_targets = targets_for_context(context)
            requested_targets = set(base_targets) | set(feature_preferences)
            if any(name in FEATURES and name not in available_features for name in feature_preferences):
                missing = sorted(name for name in feature_preferences if name in FEATURES and name not in available_features)
                raise RecommendationError(f"Requested audio feature(s) unavailable in this catalog: {', '.join(missing)}.")
            use_features = any(name in available_features for name in requested_targets)
            available = ("features", "exploration") if use_features else ("exploration",)
            context["preference"] = raw_context.get("preference", "general")
            selection = learner.choose_action(context, available_actions=available, user_id=owner)
            result = engine.recommend(
                context, limit=limit, strategy=selection["action"],
                exclude_ids=previously_recommended,
            )
            result["rotation_reset"] = False
            if not result["recommendations"] and previously_recommended:
                # Once this session has seen every matching catalog item, begin
                # a new rotation instead of leaving the user with an empty page.
                result = engine.recommend(context, limit=limit, strategy=selection["action"])
                result["rotation_reset"] = bool(result["recommendations"])
                if result["rotation_reset"]:
                    result["message"] = "You've seen all matching tracks in this session. Starting a fresh rotation."
        except RecommendationError:
            raise
        except (TypeError, ValueError) as exc:
            raise RecommendationError(str(exc)) from exc
        now = _utc_now()
        saved: list[dict[str, Any]] = []
        with connect() as conn:
            for song in result["recommendations"]:
                recommendation_id = str(uuid.uuid4())
                conn.execute("""INSERT INTO recommendations
                    (recommendation_id, user_id, song_id, context_json, strategy, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (recommendation_id, owner, str(song["song_id"]), json.dumps(context, ensure_ascii=False), selection["action"], now))
                song["recommendation_id"] = recommendation_id
                saved.append(song)
        result["recommendations"] = saved
        result["selection"] = {"action": selection["action"], "explored": selection["explored"]}
        return jsonify({"data": result}), 200

    @app.post("/api/youtube-feedback")
    def youtube_feedback():
        body = body_object()
        recommendation_id = body.get("recommendation_id")
        rating = body.get("rating")
        if not isinstance(recommendation_id, str) or not recommendation_id:
            raise RecommendationError("recommendation_id is required.")
        if not isinstance(rating, str) or rating.strip().lower() not in REWARDS:
            raise RecommendationError(f"rating must be one of: {', '.join(REWARDS)}.")
        owner = user_id()
        normalized = rating.strip().lower()
        with connect() as conn:
            row = conn.execute("SELECT video_id FROM youtube_recommendations WHERE recommendation_id=? AND user_id=?",
                               (recommendation_id, owner)).fetchone()
            if row is None:
                return error("YouTube recommendation was not found for this session.", "recommendation_not_found", 404)
            try:
                conn.execute("INSERT INTO youtube_feedback(recommendation_id,user_id,video_id,rating,reward,created_at) VALUES(?,?,?,?,?,?)",
                             (recommendation_id, owner, row["video_id"], normalized, REWARDS[normalized], _utc_now()))
            except sqlite3.IntegrityError:
                return error("Feedback was already submitted for this recommendation.", "duplicate_feedback", 409)
        return jsonify({"data": {"recommendation_id": recommendation_id, "video_id": row["video_id"],
                                 "rating": normalized, "reward": REWARDS[normalized]}}), 201

    @app.get("/api/youtube-favorites")
    def get_youtube_favorites():
        owner = user_id()
        with connect() as conn:
            rows = conn.execute("SELECT metadata_json,created_at FROM youtube_favorites WHERE user_id=? ORDER BY created_at DESC", (owner,)).fetchall()
        items = []
        for row in rows:
            item = json.loads(row["metadata_json"])
            item["favorited_at"] = row["created_at"]
            items.append(item)
        return jsonify({"data": items})

    @app.post("/api/youtube-favorites")
    def add_youtube_favorite():
        body = body_object()
        video_id = body.get("video_id")
        if not isinstance(video_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise RecommendationError("A valid YouTube video ID is required.")
        owner = user_id()
        with connect() as conn:
            row = conn.execute("SELECT result_json FROM youtube_recommendations WHERE user_id=? AND video_id=? ORDER BY created_at DESC LIMIT 1",
                               (owner, video_id)).fetchone()
            if row is None:
                return error("This YouTube result was not found in your recommendations.", "video_not_found", 404)
            metadata = json.loads(row["result_json"])
            cursor = conn.execute("INSERT OR IGNORE INTO youtube_favorites(user_id,video_id,metadata_json,created_at) VALUES(?,?,?,?)",
                                  (owner, video_id, row["result_json"], _utc_now()))
        return jsonify({"data": {"video_id": video_id, "saved": True, "already_saved": cursor.rowcount == 0}}), 201

    @app.delete("/api/youtube-favorites/<video_id>")
    def remove_youtube_favorite(video_id: str):
        owner = user_id()
        with connect() as conn:
            cursor = conn.execute("DELETE FROM youtube_favorites WHERE user_id=? AND video_id=?", (owner, video_id))
        if cursor.rowcount == 0:
            return error("Saved YouTube result was not found for this session.", "favorite_not_found", 404)
        return jsonify({"data": {"video_id": video_id, "saved": False}})

    @app.post("/api/itunes-feedback")
    def itunes_feedback():
        body = body_object()
        recommendation_id, rating = body.get("recommendation_id"), body.get("rating")
        if not isinstance(recommendation_id, str) or not recommendation_id:
            raise RecommendationError("recommendation_id is required.")
        if not isinstance(rating, str) or rating.strip().lower() not in REWARDS:
            raise RecommendationError(f"rating must be one of: {', '.join(REWARDS)}.")
        owner, normalized = user_id(), rating.strip().lower()
        with connect() as conn:
            row = conn.execute("SELECT track_id FROM itunes_recommendations WHERE recommendation_id=? AND user_id=?",
                               (recommendation_id, owner)).fetchone()
            if row is None:
                return error("Music recommendation was not found for this session.", "recommendation_not_found", 404)
            try:
                conn.execute("INSERT INTO itunes_feedback(recommendation_id,user_id,track_id,rating,reward,created_at) VALUES(?,?,?,?,?,?)",
                             (recommendation_id, owner, row["track_id"], normalized, REWARDS[normalized], _utc_now()))
            except sqlite3.IntegrityError:
                return error("Feedback was already submitted for this recommendation.", "duplicate_feedback", 409)
        return jsonify({"data": {"recommendation_id": recommendation_id, "track_id": row["track_id"],
                                 "rating": normalized, "reward": REWARDS[normalized]}}), 201

    @app.get("/api/itunes-favorites")
    def get_itunes_favorites():
        owner = user_id()
        with connect() as conn:
            rows = conn.execute("SELECT metadata_json,created_at FROM itunes_favorites WHERE user_id=? ORDER BY created_at DESC", (owner,)).fetchall()
        items = []
        for row in rows:
            item = json.loads(row["metadata_json"])
            item["favorited_at"] = row["created_at"]
            items.append(item)
        return jsonify({"data": items})

    @app.post("/api/itunes-favorites")
    def add_itunes_favorite():
        body = body_object()
        track_id = body.get("track_id")
        if not isinstance(track_id, (str, int)) or not str(track_id).isdigit():
            raise RecommendationError("A valid iTunes track ID is required.")
        track_id, owner = str(track_id), user_id()
        with connect() as conn:
            row = conn.execute("SELECT result_json FROM itunes_recommendations WHERE user_id=? AND track_id=? ORDER BY created_at DESC LIMIT 1",
                               (owner, track_id)).fetchone()
            if row is None:
                return error("This music result was not found in your recommendations.", "track_not_found", 404)
            cursor = conn.execute("INSERT OR IGNORE INTO itunes_favorites(user_id,track_id,metadata_json,created_at) VALUES(?,?,?,?)",
                                  (owner, track_id, row["result_json"], _utc_now()))
        return jsonify({"data": {"track_id": track_id, "saved": True, "already_saved": cursor.rowcount == 0}}), 201

    @app.delete("/api/itunes-favorites/<track_id>")
    def remove_itunes_favorite(track_id: str):
        if not track_id.isdigit():
            raise RecommendationError("A valid iTunes track ID is required.")
        with connect() as conn:
            cursor = conn.execute("DELETE FROM itunes_favorites WHERE user_id=? AND track_id=?", (user_id(), track_id))
        if cursor.rowcount == 0:
            return error("Saved music result was not found for this session.", "favorite_not_found", 404)
        return jsonify({"data": {"track_id": track_id, "saved": False}})

    @app.post("/api/feedback")
    def feedback():
        body = body_object()
        owner = user_id()
        recommendation_id = body.get("recommendation_id")
        rating = body.get("rating")
        if not isinstance(recommendation_id, str) or not recommendation_id:
            raise RecommendationError("recommendation_id is required.")
        if not isinstance(rating, str) or rating.strip().lower() not in REWARDS:
            raise RecommendationError(f"rating must be one of: {', '.join(REWARDS)}.")
        with connect() as conn:
            row = conn.execute("""SELECT song_id, context_json, strategy FROM recommendations
                WHERE recommendation_id=? AND user_id=?""", (recommendation_id, owner)).fetchone()
            if row is None:
                return error("Recommendation was not found for this session.", "recommendation_not_found", 404)
            context = json.loads(row["context_json"])
            reward = REWARDS[rating.strip().lower()]
            next_context = dict(context)
            next_context["feedback_summary"] = feedback_summary(conn, owner, reward)
            context["feedback_summary"] = context.get("feedback_summary", "none")
            song_id, strategy = row["song_id"], row["strategy"]
        update = learner.record_feedback(
            recommendation_id=recommendation_id,
            context=context,
            action=strategy,
            rating=rating,
            next_context=next_context,
            user_id=owner,
            song_id=song_id,
        )
        return jsonify({"data": {
            "recommendation_id": recommendation_id, "song_id": song_id,
            "rating": rating.strip().lower(), "reward": update.reward,
            "q_value_before": update.old_q, "q_value_after": update.new_q,
            "strategy": strategy,
        }}), 201

    @app.get("/api/favorites")
    def get_favorites():
        owner = user_id()
        with connect() as conn:
            rows = conn.execute("""SELECT s.*, f.created_at AS favorited_at
                FROM favorites f JOIN songs s ON s.song_id=f.song_id
                WHERE f.user_id=? ORDER BY f.created_at DESC""", (owner,)).fetchall()
        return jsonify({"data": _dict_rows(rows)})

    @app.post("/api/favorites")
    def add_favorite():
        body = body_object()
        owner = user_id()
        song_id = body.get("song_id")
        if not isinstance(song_id, str) or not song_id.strip():
            raise RecommendationError("song_id is required.")
        song_id = song_id.strip()
        with connect() as conn:
            if conn.execute("SELECT 1 FROM songs WHERE song_id=?", (song_id,)).fetchone() is None:
                return error("Song was not found in the catalog.", "song_not_found", 404)
            cursor = conn.execute("INSERT OR IGNORE INTO favorites(user_id, song_id, created_at) VALUES (?, ?, ?)",
                                  (owner, song_id, _utc_now()))
        return jsonify({"data": {"song_id": song_id, "saved": True, "already_saved": cursor.rowcount == 0}}), 201

    @app.delete("/api/favorites/<path:song_id>")
    def remove_favorite(song_id: str):
        owner = user_id()
        with connect() as conn:
            cursor = conn.execute("DELETE FROM favorites WHERE user_id=? AND song_id=?", (owner, song_id))
        if cursor.rowcount == 0:
            return error("Favorite was not found for this session.", "favorite_not_found", 404)
        return jsonify({"data": {"song_id": song_id, "saved": False}})

    @app.get("/api/history")
    def history():
        owner = user_id()
        try:
            limit = int(request.args.get("limit", "50"))
        except ValueError as exc:
            raise RecommendationError("limit must be an integer.") from exc
        if not 1 <= limit <= 100:
            raise RecommendationError("limit must be between 1 and 100.")
        with connect() as conn:
            rows = conn.execute("""SELECT r.recommendation_id, r.context_json, r.strategy,
                    r.created_at, s.*, f.rating, f.reward
                FROM recommendations r JOIN songs s ON s.song_id=r.song_id
                LEFT JOIN learning_feedback f ON f.recommendation_id=r.recommendation_id
                WHERE r.user_id=? ORDER BY r.created_at DESC LIMIT ?""", (owner, limit)).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["context"] = json.loads(item.pop("context_json"))
            items.append(item)
        with connect() as conn:
            youtube_rows = conn.execute("""SELECT r.recommendation_id,r.context_json,r.created_at,r.result_json,
                    f.rating,f.reward FROM youtube_recommendations r
                LEFT JOIN youtube_feedback f ON f.recommendation_id=r.recommendation_id
                WHERE r.user_id=? ORDER BY r.created_at DESC LIMIT ?""", (owner, limit)).fetchall()
        for row in youtube_rows:
            item = json.loads(row["result_json"])
            item.update({"context": json.loads(row["context_json"]), "recommendation_id": row["recommendation_id"],
                         "strategy": "youtube_data_api_metadata_search", "created_at": row["created_at"],
                         "rating": row["rating"], "reward": row["reward"]})
            items.append(item)
        with connect() as conn:
            itunes_rows = conn.execute("""SELECT r.recommendation_id,r.context_json,r.created_at,r.result_json,
                    f.rating,f.reward FROM itunes_recommendations r
                LEFT JOIN itunes_feedback f ON f.recommendation_id=r.recommendation_id
                WHERE r.user_id=? ORDER BY r.created_at DESC LIMIT ?""", (owner, limit)).fetchall()
        for row in itunes_rows:
            item = json.loads(row["result_json"])
            item.update({"context": json.loads(row["context_json"]), "recommendation_id": row["recommendation_id"],
                         "strategy": "itunes_search_metadata", "created_at": row["created_at"],
                         "rating": row["rating"], "reward": row["reward"]})
            items.append(item)
        items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        items = items[:limit]
        return jsonify({"data": items, "meta": {"limit": limit}})

    @app.get("/api/insights")
    def insights():
        owner = user_id()
        with connect() as conn:
            goals = conn.execute("""SELECT json_extract(context_json, '$.goal') AS goal, COUNT(*) AS recommendations
                FROM recommendations WHERE user_id=? GROUP BY goal ORDER BY recommendations DESC""", (owner,)).fetchall()
            mood_levels = conn.execute("""SELECT json_extract(context_json, '$.mood') AS mood,
                    json_extract(context_json, '$.mood_level') AS level, COUNT(*) AS selections
                FROM recommendations WHERE user_id=? GROUP BY mood, level ORDER BY selections DESC""", (owner,)).fetchall()
            feature_rows = conn.execute("""SELECT AVG(s.energy) AS energy, AVG(s.danceability) AS danceability,
                    AVG(s.acousticness) AS acousticness, AVG(s.valence) AS valence,
                    COUNT(*) AS rated_songs
                FROM learning_feedback f JOIN songs s ON s.song_id=f.song_id
                WHERE f.user_id=? AND f.reward > 0""", (owner,)).fetchone()
        learned = learner.insights(user_id=owner)
        features = dict(feature_rows) if feature_rows else {"rated_songs": 0}
        available = {key: round(float(value), 4) for key, value in features.items() if key != "rated_songs" and value is not None}
        features = {"rated_songs": int(features.get("rated_songs") or 0), "average_audio_features_on_positive_feedback": available}
        return jsonify({"data": {
            "selected_goals": _dict_rows(goals),
            "mood_levels": _dict_rows(mood_levels),
            "positive_feedback_characteristics": features,
            "feedback_and_q_learning": learned,
            "reliability_note": "Summaries reflect only this session's stored interactions; small samples are descriptive, not statistically reliable.",
        }})

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="127.0.0.1", port=int(os.environ.get("TUNELY_PORT", "5002")), debug=False)
