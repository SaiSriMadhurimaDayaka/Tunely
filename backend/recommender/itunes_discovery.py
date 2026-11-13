"""No-key iTunes Search API discovery with conservative language evidence."""

from __future__ import annotations

import json
import logging
import re
import socket
import ssl
from datetime import date
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from backend.recommender.youtube_discovery import GOAL_SEARCH_TERMS, MOOD_PROFILES, SUPPORTED_LANGUAGES, mood_genres, normalize_intensity
from backend.recommender.external_links import youtube_music_search_url
from backend.recommender.music_categories import CATEGORY_SEARCH_TERMS, classify_music_categories, normalize_category


class ITunesDiscoveryError(RuntimeError):
    def __init__(self, message: str, code: str = "itunes_api_error"):
        super().__init__(message)
        self.code = code


class ITunesSearchClient:
    endpoint = "https://itunes.apple.com/search"

    def __init__(self, *, timeout: float = 8.0, opener=urlopen):
        self.timeout = timeout
        self.opener = opener

    def search(self, term: str, country: str = "in", limit: int = 50) -> list[dict[str, Any]]:
        query = urlencode({"term": term, "entity": "song", "media": "music", "country": country, "limit": limit})
        request = Request(f"{self.endpoint}?{query}", headers={"Accept": "application/json", "User-Agent": "Tunely/1.0"})
        try:
            with self.opener(request, timeout=self.timeout) as response:
                response_body = response.read()
        except HTTPError as exc:
            logging.exception("iTunes Search API returned HTTP %s for term %r", exc.code, term)
            exc.close()
            if exc.code == 429:
                raise ITunesDiscoveryError("The music catalog is receiving too many requests. Please try again shortly.", "rate_limited") from exc
            raise ITunesDiscoveryError(f"The music catalog returned HTTP {exc.code}. Please try again later.", "upstream_error") from exc
        except (URLError, TimeoutError, OSError) as exc:
            logging.exception("iTunes Search API request failed for term %r", term)
            reason = exc.reason if isinstance(exc, URLError) else exc
            reason_text = str(reason)
            if isinstance(reason, PermissionError):
                raise ITunesDiscoveryError(
                    "This Python process is not permitted to open a network connection to iTunes. Check its firewall, sandbox, or network policy.",
                    "network_restricted",
                ) from exc
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise ITunesDiscoveryError(
                    f"The iTunes Search API request timed out after {self.timeout:g} seconds. Please retry.",
                    "timeout",
                ) from exc
            if isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in reason_text:
                raise ITunesDiscoveryError(
                    "Python could not verify iTunes' HTTPS certificate. Check the Python certificate store or HTTPS inspection proxy.",
                    "tls_error",
                ) from exc
            raise ITunesDiscoveryError(
                f"Could not connect to iTunes ({type(reason).__name__}: {reason_text}). See the Flask log for details.",
                "network_error",
            ) from exc
        except (ValueError, UnicodeDecodeError) as exc:
            logging.exception("iTunes Search API returned invalid JSON for term %r", term)
            raise ITunesDiscoveryError("The music catalog returned an unreadable response.", "invalid_response") from exc
        try:
            payload = json.loads(response_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            logging.exception("iTunes Search API returned invalid JSON for term %r", term)
            raise ITunesDiscoveryError("The music catalog returned an unreadable response.", "invalid_response") from exc
        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            logging.error("iTunes Search API response did not contain a results array for term %r", term)
            raise ITunesDiscoveryError("The music catalog returned an unreadable response.", "invalid_response")
        return [item for item in results if isinstance(item, dict) and item.get("kind") == "song"]


_LANGUAGE_STOPWORDS = {
    "English": {"the", "and", "you", "your", "with", "love", "for", "is", "my", "me", "are", "we", "of", "to", "in"},
    "French": {"les", "des", "une", "est", "avec", "pour", "mon", "ma", "toi", "amour", "que", "dans", "je", "la"},
    "Spanish": {"los", "las", "una", "con", "para", "mi", "amor", "que", "del", "por", "yo", "de", "el"},
}
_EXPLICIT_LANGUAGE = {
    "Arabic": ("arabic", "عربي", "العربية"), "Bengali": ("bengali", "bangla", "বাংলা"),
    "English": ("english",), "French": ("french", "français"),
    "Gujarati": ("gujarati", "ગુજરાતી"), "Hindi": ("hindi", "हिन्दी", "हिंदी"),
    "Japanese": ("japanese", "日本語"), "Kannada": ("kannada", "ಕನ್ನಡ"),
    "Korean": ("korean", "한국어"), "Malayalam": ("malayalam", "മലയാളം"),
    "Marathi": ("marathi", "मराठी"), "Punjabi": ("punjabi", "ਪੰਜਾਬੀ"),
    "Spanish": ("spanish", "español"), "Tamil": ("tamil", "தமிழ்"),
    "Telugu": ("telugu", "తెలుగు"), "Urdu": ("urdu", "اردو"),
}
_SCRIPT_ONLY = {"Bengali", "Gujarati", "Japanese", "Kannada", "Korean", "Malayalam", "Punjabi", "Tamil", "Telugu"}


def _language_confidence(text: str, language: str) -> tuple[float, str]:
    folded = text.casefold()
    aliases = _EXPLICIT_LANGUAGE.get(language, ())
    if any(re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", folded) for alias in aliases):
        return 1.0, "explicit_language_label_in_catalog_text"
    script = SUPPORTED_LANGUAGES[language]["script"]
    if language in _SCRIPT_ONLY and re.search(script, text):
        return 0.90, "distinct_writing_system_in_catalog_text"
    words = set(re.findall(r"[a-zÀ-ÿ']+", folded))
    target = _LANGUAGE_STOPWORDS.get(language, set())
    hits = len(words & target)
    other_hits = max((len(words & values) for key, values in _LANGUAGE_STOPWORDS.items() if key != language), default=0)
    if hits >= 2 and hits > other_hits:
        return 0.72, "language_wording_estimate"
    return 0.0, "language_not_reliably_identifiable_from_catalog_metadata"


def build_itunes_term(language: str, mood: str, intensity: str, goal: str = "",
                      category: str | None = None) -> tuple[str, list[str]]:
    level, normalized = normalize_intensity(intensity)
    if not level or mood not in MOOD_PROFILES:
        raise ValueError("A supported mood and intensity are required.")
    genres = mood_genres(mood, normalized)
    # iTunes treats terms as broad catalog text matches: adding mood/genre words
    # can swamp the requested language and return generic tracks. Search the
    # selected language first; mood, intensity, and automatic genres rank and
    # explain the returned metadata below.
    category_id = normalize_category(category)
    term = f"{language} {CATEGORY_SEARCH_TERMS[category_id]}" if category_id else f"{language} songs"
    return term, genres


def build_itunes_query_variants(language: str, mood: str, intensity: str, goal: str = "",
                                preferred_categories: Sequence[str] = ()) -> list[str]:
    """Build deterministic, distinct provider searches from all chosen inputs."""
    level, normalized = normalize_intensity(intensity)
    if not level or mood not in MOOD_PROFILES:
        raise ValueError("A supported mood and intensity are required.")
    genres = mood_genres(mood, normalized)
    profile = MOOD_PROFILES[mood]
    intensity_term = profile["intensity"][level - 1]
    variants = [f"{language} {intensity_term} songs"]
    if genres:
        variants.append(f"{language} {genres[0]} songs")
    if goal in GOAL_SEARCH_TERMS:
        variants.append(f"{language} {GOAL_SEARCH_TERMS[goal]} songs")
    for category in preferred_categories:
        category_term = CATEGORY_SEARCH_TERMS.get(category)
        if category_term:
            variants.append(f"{language} {category_term}")
    return list(dict.fromkeys(term.strip() for term in variants if term.strip()
                              and term.casefold() != f"{language} songs".casefold()))


def build_itunes_seed_query(language: str, items: Sequence[Mapping[str, Any]], *,
                            recent_artists: Sequence[str] = (), recent_albums: Sequence[str] = (),
                            rotation: int = 0) -> str | None:
    """Use actual returned artist/album metadata for an additional search seam."""
    recent_artist_keys = {str(value).strip().casefold() for value in recent_artists if value}
    recent_album_keys = {str(value).strip().casefold() for value in recent_albums if value}
    artists: dict[str, str] = {}
    albums: dict[str, str] = {}
    for item in items:
        artist = str(item.get("artistName") or "").strip()
        album = str(item.get("collectionName") or "").strip()
        if artist and len(artist) <= 70 and artist.casefold() not in recent_artist_keys:
            artists.setdefault(artist.casefold(), artist)
        if album and len(album) <= 70 and album.casefold() not in recent_album_keys:
            albums.setdefault(album.casefold(), album)
    seeds = [*(f"{language} {value}" for value in artists.values()), *(f"{language} {value}" for value in albums.values())]
    return seeds[rotation % len(seeds)] if seeds else None


def rank_itunes_results(items: Sequence[Mapping[str, Any]], *, language: str, mood: str,
                        intensity: str, genres: Sequence[str], limit: int = 8,
                        goal: str = "", category: str | None = None,
                        preferred_categories: Sequence[str] = (),
                        recent_track_ids: Sequence[str] = (),
                        exclude_track_ids: Sequence[str] = (),
                        recent_artists: Sequence[str] = (),
                        recent_albums: Sequence[str] = ()) -> list[dict[str, Any]]:
    category_id = normalize_category(category)
    profile = MOOD_PROFILES[mood]
    intensity_index = normalize_intensity(intensity)[0] - 1
    mood_terms = profile["terms"]
    intensity_terms = re.findall(r"[a-z]+", profile["intensity"][intensity_index].casefold())
    goal_terms = re.findall(r"[a-z]+", GOAL_SEARCH_TERMS.get(goal, "").casefold())
    candidates = []
    seen = set()
    recent_ids = {str(value) for value in recent_track_ids}
    excluded_ids = {str(value) for value in exclude_track_ids}
    recent_artist_keys = {str(value).strip().casefold() for value in recent_artists if value}
    recent_album_keys = {str(value).strip().casefold() for value in recent_albums if value}
    for position, item in enumerate(items):
        title = str(item.get("trackName") or "").strip()
        artist = str(item.get("artistName") or "").strip()
        if not title:
            continue
        track_id = item.get("trackId")
        if not isinstance(track_id, (int, str)) or not str(track_id).isdigit():
            continue
        key = str(track_id)
        if key in seen:
            continue
        seen.add(key)
        collection = str(item.get("collectionName") or "")
        provider_genre = str(item.get("primaryGenreName") or "").strip()
        classification = classify_music_categories(item)
        if category_id and category_id not in classification["category_ids"]:
            continue
        evidence_text = " ".join((title, artist, collection, provider_genre))
        language_confidence, language_evidence = _language_confidence(evidence_text, language)
        folded = evidence_text.casefold()
        mood_hits = [term for term in mood_terms if term in folded]
        intensity_hits = [term for term in intensity_terms if term in folded]
        goal_hits = [term for term in goal_terms if term in folded]
        provider_genre_match = any(term.casefold() in provider_genre.casefold() for term in genres)
        preference_matches = set(preferred_categories) & set(classification["category_ids"])
        preference_score = 0.08 * min(1, len(preference_matches))
        score = (0.50 * language_confidence + 0.13 * min(1.0, len(mood_hits) * 0.35)
                 + 0.16 * min(1.0, len(intensity_hits) * 0.35)
                 + 0.10 * min(1.0, len(goal_hits) * 0.5)
                 + 0.08 * (1.0 if provider_genre_match else 0.35 if provider_genre else 0.0)
                 + 0.03 * max(0.0, 1 - position / max(1, len(items)))
                 + (0.05 * classification["category_scores"].get(category_id, 0.0) if category_id else preference_score))
        if mood == "nostalgic":
            if "old" in classification["category_ids"]:
                score += 0.20
            elif "latest" in classification["category_ids"]:
                score -= 0.25
        artwork = item.get("artworkUrl100") or item.get("artworkUrl60")
        if isinstance(artwork, str) and artwork.startswith("http://"):
            artwork = "https://" + artwork[len("http://"):]
        if not isinstance(artwork, str) or not artwork.startswith("https://"):
            artwork = None
        track_url = item.get("trackViewUrl")
        if not isinstance(track_url, str) or not track_url.startswith("https://"):
            track_url = None
        raw_release_date = item.get("releaseDate")
        release_year = None
        if isinstance(raw_release_date, str):
            try:
                release_year = str(date.fromisoformat(raw_release_date[:10]).year)
            except ValueError:
                pass
        assigned_genre = genres[0] if genres else provider_genre or None
        emotion = item.get("_tunely_emotion") if isinstance(item.get("_tunely_emotion"), Mapping) else {}
        reason_evidence = []
        if language_confidence:
            reason_evidence.append(f"catalog text provides {language_evidence.replace('_', ' ')}")
        else:
            reason_evidence.append(f"the search targeted {language}, but catalog metadata does not verify the track language")
        if mood_hits:
            reason_evidence.append("title or album metadata includes mood-related wording")
        elif provider_genre_match:
            reason_evidence.append("the catalog genre aligns with an automatically selected mood style")
        else:
            reason_evidence.append(f"the selected {intensity} {mood} mood maps to the automatic {genres[0]} style")
        if goal:
            reason_evidence.append("catalog text also matches the listening goal" if goal_hits
                                   else f"the {goal} listening goal shaped the additional catalog search")
        candidates.append({
            "song_id": f"itunes_{key}", "itunes_track_id": key, "song_name": title, "title": title,
            "artist": artist, "album_name": collection or None, "release_year": release_year,
            "duration_ms": item.get("trackTimeMillis"),
            "language": language if language_confidence else "Unknown", "requested_language": language,
            "language_verified": False,
            "language_confidence": language_confidence, "language_evidence": language_evidence,
            "genre": provider_genre or None, "genre_candidates": list(genres), "ai_genre": assigned_genre,
            "itunes_genre": provider_genre or None, "mood_evidence": mood_hits, "intensity_evidence": intensity_hits,
            "goal_evidence": goal_hits,
            "selected_mood": mood, "mood_intensity": intensity, "mood_verified": False,
            "cover_image": artwork, "itunes_track_url": track_url,
            "emotion_labels": list(emotion.get("labels") or []),
            "emotion_confidence": dict(emotion.get("confidence") or {}),
            "emotion_evidence": dict(emotion.get("evidence") or {}),
            "emotion_status": "classified" if emotion.get("labels") else "unknown",
            "emotion_model_version": emotion.get("model_version", "not_classified"),
            "emotion_confidence_kind": emotion.get("confidence_kind", "unknown"),
            "lyrics_availability": emotion.get("lyrics_availability", "unknown"),
            "lyrics_lookup_status": emotion.get("lyrics_lookup_status", "not_checked"),
            "audio_analysis_status": emotion.get("audio_analysis_status", "unavailable_no_authorized_audio"),
            **classification,
            "mood_category_preferences": list(preferred_categories),
            "youtube_music_search_url": youtube_music_search_url(title, artist),
            "score": round(score, 5), "strategy": "itunes_search_metadata",
            "explanation": f"Suggested for {intensity} {mood} because " + "; and ".join(reason_evidence) + ".",
            "_recent": key in recent_ids,
            "_currently_displayed": key in excluded_ids,
        })
    # A dated recent release is direct evidence against an ordinary nostalgia
    # request. Keep undated tracks eligible, but never present a track explicitly
    # labeled Latest unless the user also asks to discover something new.
    if mood == "nostalgic" and goal != "discover something new":
        candidates = [row for row in candidates if "latest" not in row["category_ids"]]
    current_cooldown = [row for row in candidates if not row["_recent"] and not row["_currently_displayed"]]
    pool = list(current_cooldown)
    # Reuse tracks only after exhausting unshown, non-recent alternatives.
    for row in candidates:
        if len(pool) >= limit:
            break
        if row not in pool and not row["_currently_displayed"]:
            pool.append(row)
    for row in candidates:
        if len(pool) >= limit:
            break
        if row not in pool:
            pool.append(row)

    artist_counts: dict[str, int] = {}
    album_counts: dict[str, int] = {}
    year_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    while pool and len(selected) < limit:
        def diverse_score(row: Mapping[str, Any]) -> float:
            artist_key = str(row.get("artist") or "").strip().casefold()
            album_key = str(row.get("album_name") or "").strip().casefold()
            year_key = str(row.get("release_year") or "")
            repeat_penalty = (0.12 * artist_counts.get(artist_key, 0)
                              + 0.07 * album_counts.get(album_key, 0)
                              + 0.025 * year_counts.get(year_key, 0))
            history_penalty = (0.04 if artist_key and artist_key in recent_artist_keys else 0.0)
            history_penalty += 0.02 if album_key and album_key in recent_album_keys else 0.0
            history_penalty += 0.18 if row["_recent"] else 0.0
            return float(row["score"]) - repeat_penalty - history_penalty

        unique_artist_pool = [row for row in pool if not row.get("artist")
                              or str(row["artist"]).strip().casefold() not in artist_counts]
        artist_pool = unique_artist_pool or pool
        unique_album_pool = [row for row in artist_pool if not row.get("album_name")
                             or str(row["album_name"]).strip().casefold() not in album_counts]
        diverse_pool = unique_album_pool or artist_pool
        best = max(diverse_pool, key=lambda row: (diverse_score(row), row["score"], row["itunes_track_id"]))
        pool.remove(best)
        best["score"] = round(max(0.0, diverse_score(best)), 5)
        artist_key = str(best.get("artist") or "").strip().casefold()
        album_key = str(best.get("album_name") or "").strip().casefold()
        year_key = str(best.get("release_year") or "")
        if artist_key:
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        if album_key:
            album_counts[album_key] = album_counts.get(album_key, 0) + 1
        if year_key:
            year_counts[year_key] = year_counts.get(year_key, 0) + 1
        best.pop("_recent", None)
        best.pop("_currently_displayed", None)
        selected.append(best)
    return selected
