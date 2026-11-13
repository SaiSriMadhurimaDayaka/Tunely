"""LRCLIB lookup, no-lyrics-on-disk cache, and multilingual emotion persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from backend.recommender.emotion_analysis import EMOTION_MODEL_VERSION, LocalXLMREmotionClassifier
from backend.recommender.lyrics_provider import LRCLibClient, LyricsProviderError


def _identity_key(song: Mapping[str, Any], language: str, model_version: str) -> str:
    parts = [song.get("song_name") or song.get("trackName") or song.get("title"),
             song.get("artist") or song.get("artistName"), song.get("album_name") or song.get("collectionName"),
             song.get("duration_ms") or song.get("trackTimeMillis"), language, model_version]
    normalized = "|".join(" ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split()) for value in parts)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def enrich_song_emotion(conn, song: dict[str, Any], language: str, *,
                        provider: LRCLibClient | None = None,
                        classifier: LocalXLMREmotionClassifier | None = None,
                        now: datetime | None = None) -> dict[str, Any]:
    """Enrich a recommendation from matched lyrics, persisting only analysis metadata."""
    now = now or datetime.now(timezone.utc)
    current = now.isoformat(timespec="seconds")
    engine = classifier or LocalXLMREmotionClassifier()
    model_signature = (f"xlmr-local:{os.path.basename(engine.model_path)}"
                       if getattr(engine, "available", False) else EMOTION_MODEL_VERSION)
    key = _identity_key(song, language, model_signature)
    cached = conn.execute("SELECT * FROM itunes_lyrics_cache WHERE query_key=? AND expires_at>=?", (key, current)).fetchone()
    if cached:
        analysis = {
            "lyrics_availability": "available" if cached["lyrics_available"] else "unavailable",
            "lyrics_lookup_status": cached["lookup_status"],
            "audio_analysis_status": "unavailable_no_authorized_audio",
            "labels": json.loads(cached["emotion_labels_json"] or "[]"),
            "confidence": json.loads(cached["emotion_confidence_json"] or "{}"),
            "evidence": json.loads(cached["emotion_evidence_json"] or "{}"),
            "model_version": cached["emotion_model_version"],
            "confidence_kind": cached["emotion_confidence_kind"],
            "lyrics_match_score": cached["match_score"],
        }
        analysis["status"] = "classified" if analysis["labels"] else "unknown"
    else:
        analysis = {"lyrics_availability": "unavailable", "lyrics_lookup_status": "unknown",
                    "audio_analysis_status": "unavailable_no_authorized_audio", "labels": [],
                    "confidence": {}, "evidence": {}, "model_version": "not_classified",
                    "confidence_kind": "unknown", "lyrics_match_score": None}
        lookup_record = {}
        retry_ttl_days = 10 / 1440
        try:
            source = provider or LRCLibClient(timeout=4.0)
            lookup_record = source.lookup(song)
            analysis["lyrics_lookup_status"] = lookup_record.get("status", "unknown")
            analysis["lyrics_availability"] = "available" if lookup_record.get("lyrics_available") else "unavailable"
            analysis["lyrics_match_score"] = lookup_record.get("match_score")
            if lookup_record.get("lyrics_available"):
                result = engine.classify(lookup_record.get("lyrics_text", ""), language)
                analysis.update({key: result.get(key, analysis.get(key)) for key in
                                 ("labels", "confidence", "evidence", "status", "model_version", "confidence_kind")})
        except LyricsProviderError as exc:
            analysis["lyrics_lookup_status"] = exc.code
            analysis["provider_error"] = str(exc)
            if exc.retry_after:
                retry_ttl_days = max(retry_ttl_days, exc.retry_after / 86400)
        except Exception:
            logging.exception("Unexpected lyrics analysis failure for iTunes track %r", song.get("itunes_track_id"))
            analysis["lyrics_lookup_status"] = "analysis_error"
            analysis["provider_error"] = "Lyrics analysis was unavailable for this track."

        success = analysis["lyrics_availability"] == "available"
        error = analysis["lyrics_lookup_status"] in {
            "lyrics_http_error", "lyrics_network_error", "lyrics_invalid_response", "analysis_error",
            "lyrics_rate_limited", "lyrics_server_busy",
        }
        ttl = 14 if success or not error else retry_ttl_days
        expires = (now + timedelta(days=ttl)).isoformat(timespec="seconds")
        conn.execute("""INSERT INTO itunes_lyrics_cache
            (query_key,lookup_status,lyrics_available,lrclib_record_id,matched_title,matched_artist,match_score,
             emotion_labels_json,emotion_confidence_json,emotion_evidence_json,emotion_model_version,
             emotion_confidence_kind,checked_at,expires_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(query_key) DO UPDATE SET
            lookup_status=excluded.lookup_status,lyrics_available=excluded.lyrics_available,
            lrclib_record_id=excluded.lrclib_record_id,matched_title=excluded.matched_title,
            matched_artist=excluded.matched_artist,match_score=excluded.match_score,
            emotion_labels_json=excluded.emotion_labels_json,emotion_confidence_json=excluded.emotion_confidence_json,
            emotion_evidence_json=excluded.emotion_evidence_json,emotion_model_version=excluded.emotion_model_version,
            emotion_confidence_kind=excluded.emotion_confidence_kind,checked_at=excluded.checked_at,expires_at=excluded.expires_at""",
            (key, analysis["lyrics_lookup_status"], int(success), lookup_record.get("record_id"),
             lookup_record.get("matched_title"), lookup_record.get("matched_artist"), analysis["lyrics_match_score"],
             json.dumps(analysis["labels"], ensure_ascii=False), json.dumps(analysis["confidence"], ensure_ascii=False),
             json.dumps(analysis["evidence"], ensure_ascii=False), analysis["model_version"],
             analysis["confidence_kind"], current, expires))

    track_id = str(song.get("itunes_track_id") or song.get("trackId") or "")
    if track_id.isdigit():
        conn.execute("""UPDATE itunes_tracks SET lyrics_availability=?,lyrics_lookup_status=?,audio_analysis_status=?,
            emotion_labels_json=?,emotion_confidence_json=?,emotion_evidence_json=?,emotion_model_version=?,
            emotion_confidence_kind=? WHERE track_id=?""",
            (analysis["lyrics_availability"], analysis["lyrics_lookup_status"], analysis["audio_analysis_status"],
             json.dumps(analysis["labels"], ensure_ascii=False), json.dumps(analysis["confidence"], ensure_ascii=False),
             json.dumps(analysis["evidence"], ensure_ascii=False), analysis["model_version"],
             analysis["confidence_kind"], track_id))
    song.update({
        "emotion_labels": analysis["labels"], "emotion_confidence": analysis["confidence"],
        "emotion_evidence": analysis["evidence"], "emotion_status": analysis.get("status", "unknown"),
        "emotion_model_version": analysis["model_version"],
        "emotion_confidence_kind": analysis["confidence_kind"],
        "lyrics_availability": analysis["lyrics_availability"],
        "lyrics_lookup_status": analysis["lyrics_lookup_status"],
        "lyrics_match_score": analysis.get("lyrics_match_score"),
        "audio_analysis_status": analysis["audio_analysis_status"],
    })
    if analysis["labels"]:
        label_text = ", ".join(analysis["labels"])
        song["explanation"] = (song.get("explanation", "").rstrip(".")
                                + f" Lyrics analysis found evidence for {label_text.lower()}.")
    if "provider_error" in analysis:
        song["emotion_note"] = analysis["provider_error"]
    return song
