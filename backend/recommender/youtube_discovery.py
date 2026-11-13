"""Language-aware, explainable YouTube Data API music discovery.

The module searches video metadata only. It does not scrape YouTube Music or
download/play audio. Language confidence values are evidence scores from
explicit labels or writing-system matches; they are not calibrated probabilities.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen


SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "Arabic": {"code": "ar", "aliases": "Arabic|عربي|العربية", "script": r"[\u0600-\u06FF]"},
    "Bengali": {"code": "bn", "aliases": "Bengali|Bangla|বাংলা", "script": r"[\u0980-\u09FF]"},
    "English": {"code": "en", "aliases": "English", "script": r"[A-Za-z]"},
    "French": {"code": "fr", "aliases": "French|Français", "script": r"[A-Za-zÀ-ÖØ-öø-ÿ]"},
    "Gujarati": {"code": "gu", "aliases": "Gujarati|ગુજરાતી", "script": r"[\u0A80-\u0AFF]"},
    "Hindi": {"code": "hi", "aliases": "Hindi|हिन्दी|हिंदी", "script": r"[\u0900-\u097F]"},
    "Japanese": {"code": "ja", "aliases": "Japanese|日本語", "script": r"[\u3040-\u30FF\u4E00-\u9FFF]"},
    "Kannada": {"code": "kn", "aliases": "Kannada|ಕನ್ನಡ", "script": r"[\u0C80-\u0CFF]"},
    "Korean": {"code": "ko", "aliases": "Korean|한국어", "script": r"[\uAC00-\uD7AF]"},
    "Malayalam": {"code": "ml", "aliases": "Malayalam|മലയാളം", "script": r"[\u0D00-\u0D7F]"},
    "Marathi": {"code": "mr", "aliases": "Marathi|मराठी", "script": r"[\u0900-\u097F]"},
    "Punjabi": {"code": "pa", "aliases": "Punjabi|ਪੰਜਾਬੀ", "script": r"[\u0A00-\u0A7F]"},
    "Spanish": {"code": "es", "aliases": "Spanish|Español", "script": r"[A-Za-zÀ-ÖØ-öø-ÿ]"},
    "Tamil": {"code": "ta", "aliases": "Tamil|தமிழ்", "script": r"[\u0B80-\u0BFF]"},
    "Telugu": {"code": "te", "aliases": "Telugu|తెలుగు", "script": r"[\u0C00-\u0C7F]"},
    "Urdu": {"code": "ur", "aliases": "Urdu|اردو", "script": r"[\u0600-\u06FF]"},
}

MOOD_PROFILES: dict[str, dict[str, Any]] = {
    "happy": {"genres": ["upbeat pop", "dance", "cheerful acoustic"], "terms": ["happy", "cheerful", "uplifting", "joyful"], "intensity": ["light feel good", "lively", "celebration energetic"]},
    "sad": {"genres": ["emotional melody", "ballad", "acoustic"], "terms": ["sad", "emotional", "melancholy", "reflective"], "intensity": ["gentle reflective", "emotional expressive", "deeply emotional cathartic"]},
    "calm": {"genres": ["ambient", "soft instrumental", "mellow acoustic"], "terms": ["calm", "peaceful", "soothing", "relaxing"], "intensity": ["soft peaceful", "soothing mellow", "tranquil meditative"]},
    "energetic": {"genres": ["dance", "rock", "upbeat pop"], "terms": ["energetic", "upbeat", "lively", "powerful"], "intensity": ["light upbeat", "lively rhythm", "high energy powerful"]},
    "stressed": {"genres": ["soothing acoustic", "gentle instrumental", "mellow"], "terms": ["soothing", "gentle", "unwind", "calm"], "intensity": ["gentle low intensity", "calm and engaging", "strong response"]},
    "romantic": {"genres": ["romantic ballad", "soft melody", "acoustic"], "terms": ["romantic", "love", "intimate", "tender"], "intensity": ["tender", "warm intimate", "expressive passionate"]},
    "neutral": {"genres": ["easy listening", "mellow pop", "acoustic"], "terms": ["mellow", "balanced", "easy listening"], "intensity": ["light", "balanced", "pronounced"]},
    "motivated": {"genres": ["uplifting pop", "rock", "motivational"], "terms": ["motivational", "inspiring", "uplifting", "powerful"], "intensity": ["encouraging", "driving", "bold powerful"]},
    "nostalgic": {"genres": ["classic melodies", "retro pop", "oldies"], "terms": ["nostalgic", "retro", "classic", "memories"], "intensity": ["familiar", "warm evocative", "deeply evocative"]},
}

# The weights sum to 1. Search position is an API-provided relevance signal;
# absent song evidence receives lower component scores, never fabricated labels.
SCORING_WEIGHTS = {"language": 0.50, "mood_evidence": 0.15, "intensity_evidence": 0.10,
                   "genre_evidence": 0.10, "metadata_completeness": 0.05, "search_position": 0.10}
MIN_LANGUAGE_EVIDENCE = 0.50
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
GOAL_SEARCH_TERMS = {
    "relax": "relax and unwind", "focus": "focus study", "energize": "energize",
    "feel-good": "feel good", "wind down": "wind down", "discover something new": "new music discovery",
}


class YouTubeDiscoveryError(RuntimeError):
    def __init__(self, message: str, code: str = "youtube_api_error"):
        super().__init__(message)
        self.code = code


def normalize_intensity(value: Any) -> tuple[int, str]:
    labels = {"mild": (1, "mild"), "moderate": (2, "moderate"), "intense": (3, "intense")}
    if isinstance(value, int) and not isinstance(value, bool):
        return {1: (1, "mild"), 2: (2, "moderate"), 3: (3, "intense")}.get(value, (0, ""))
    return labels.get(str(value or "").strip().casefold(), (0, ""))


def mood_genres(mood: str, intensity: str) -> list[str]:
    profile = MOOD_PROFILES[mood]
    genres = list(profile["genres"])
    if intensity == "intense" and mood in {"energetic", "happy", "motivated"}:
        genres.insert(0, "high energy " + genres[0])
    elif intensity == "mild" and mood in {"sad", "stressed", "calm", "romantic"}:
        genres.insert(0, "gentle " + genres[0])
    return list(dict.fromkeys(genres))


def build_search_query(language: str, mood: str, intensity: str, genres: Sequence[str], *,
                       refined: bool = False, goal: str = "") -> str:
    profile = MOOD_PROFILES[mood]
    style = profile["intensity"][{"mild": 0, "moderate": 1, "intense": 2}[intensity]]
    parts = [language, style, *genres[:2], GOAL_SEARCH_TERMS.get(goal, ""), "songs"]
    if refined:
        parts.extend(("official audio", "music video"))
    return " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))


def language_evidence(text: str, language: str) -> tuple[float, str]:
    """Return a deterministic heuristic evidence score and its evidence type."""
    config = SUPPORTED_LANGUAGES[language]
    normalized = text.casefold()
    aliases = [alias.strip().casefold() for alias in config["aliases"].split("|")]
    if any(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized) for alias in aliases):
        return 0.78, "explicit_language_label_in_video_text"
    if re.search(config["script"], text):
        # A Latin title is common across English, French, and Spanish; it cannot
        # distinguish them. Devanagari and Arabic are also shared by languages.
        if language in {"English", "French", "Spanish"}:
            return 0.45, "latin_script_common_to_multiple_languages"
        return (0.60 if language in {"Hindi", "Marathi", "Arabic", "Urdu"} else 0.90), "writing_system_detected_in_video_text"
    if language == "English" and re.search(r"[A-Za-z]{3,}", text):
        return 0.52, "latin_script_only_language_unverified"
    return 0.25, "search_query_only_language_unverified"


def _evidence(text: str, terms: Sequence[str], default: float) -> tuple[float, list[str]]:
    folded = text.casefold()
    matched = [term for term in terms if term.casefold() in folded]
    return (min(1.0, 0.55 + 0.15 * len(matched)), matched) if matched else (default, [])


def rank_candidates(items: Sequence[Mapping[str, Any]], *, language: str, mood: str,
                    intensity: str, genres: Sequence[str], limit: int = 8,
                    goal: str = "") -> list[dict[str, Any]]:
    profile = MOOD_PROFILES[mood]
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()
    for position, item in enumerate(items):
        video_id = item.get("video_id")
        title = str(item.get("title") or "").strip()
        channel = str(item.get("channel_title") or "").strip()
        description = str(item.get("description") or "")
        if not isinstance(video_id, str) or not VIDEO_ID_PATTERN.fullmatch(video_id) or not title:
            continue
        title_key = (re.sub(r"[^\w]+", " ", title.casefold()).strip(), re.sub(r"[^\w]+", " ", channel.casefold()).strip())
        if video_id in seen_ids or title_key in seen_titles:
            continue
        seen_ids.add(video_id)
        seen_titles.add(title_key)
        text = " ".join((title, description, channel))
        lang_confidence, language_method = language_evidence(text, language)
        mood_confidence, mood_hits = _evidence(text, profile["terms"], 0.35)
        intensity_confidence, intensity_hits = _evidence(text, [profile["intensity"][{"mild": 0, "moderate": 1, "intense": 2}[intensity]]], 0.30)
        genre_confidence, genre_hits = _evidence(text, list(genres), 0.35)
        metadata_completeness = (0.5 if title else 0) + (0.3 if channel else 0) + (0.2 if item.get("thumbnail_url") else 0)
        search_position = max(0.0, 1.0 - position / max(1, len(items)))
        score = (SCORING_WEIGHTS["language"] * lang_confidence
                 + SCORING_WEIGHTS["mood_evidence"] * mood_confidence
                 + SCORING_WEIGHTS["intensity_evidence"] * intensity_confidence
                 + SCORING_WEIGHTS["genre_evidence"] * genre_confidence
                 + SCORING_WEIGHTS["metadata_completeness"] * metadata_completeness
                 + SCORING_WEIGHTS["search_position"] * search_position)
        if lang_confidence < MIN_LANGUAGE_EVIDENCE:
            continue
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        candidates.append({
            "song_id": f"yt_{video_id}", "youtube_video_id": video_id,
            "song_name": title, "artist": None, "channel_title": channel or None,
            "description": description or None, "language": language,
            "language_confidence": lang_confidence, "language_verified": False,
            "language_evidence": language_method,
            "genres": genre_hits, "genre_candidates": list(genres),
            "mood_evidence": mood_hits, "intensity_evidence": intensity_hits,
            "genre_confidence": genre_confidence, "intensity_confidence": intensity_confidence,
            "mood": None, "selected_mood": mood,
            "mood_verified": False,
            "mood_confidence": mood_confidence, "mood_intensity": intensity,
            "cover_image": item.get("thumbnail_url"),
            "source_url": video_url,
            "youtube_music_url": f"https://music.youtube.com/watch?v={video_id}",
            "youtube_music_search_url": f"https://music.youtube.com/search?{urlencode({'q': build_search_query(language, mood, intensity, genres, goal=goal)})}",
            "score": round(score, 5), "strategy": "youtube_data_api_metadata_search",
            "explanation": _explanation(language, mood, intensity, language_method, mood_hits,
                                         intensity_hits, genre_hits, goal),
            "metadata_source": "YouTube Data API v3 search.list",
            "last_verified": item.get("last_verified") or item.get("last_retrieved_at"),
        })
    candidates.sort(key=lambda row: (row["score"], row["youtube_video_id"]), reverse=True)
    return candidates[:limit]


def _explanation(language: str, mood: str, intensity: str, language_method: str,
                 mood_hits: Sequence[str], intensity_hits: Sequence[str], genre_hits: Sequence[str],
                 goal: str = "") -> str:
    evidence = []
    if language_method == "writing_system_detected_in_video_text":
        evidence.append("the title or description uses a writing system associated with the selected language")
    elif language_method == "explicit_language_label_in_video_text":
        evidence.append("the video text explicitly mentions the selected language")
    else:
        evidence.append("the result came from a search for the selected language; the song language is unverified")
    if mood_hits:
        evidence.append("video text includes mood-related terms")
    if intensity_hits:
        evidence.append("video text includes terms related to the selected intensity")
    if genre_hits:
        evidence.append("video text includes suggested style terms")
    reason = f"Found for {intensity} {mood} {language} discovery because " + "; and ".join(evidence)
    if goal:
        reason += f"; the {goal} goal was included in the search"
    return reason + "."


class YouTubeDataAPIClient:
    """Small standard-library client for the documented YouTube Data API."""

    endpoint = "https://www.googleapis.com/youtube/v3/search"

    def __init__(self, api_key: str, *, timeout: float = 8.0, opener=urlopen):
        self.api_key = api_key
        self.timeout = timeout
        self.opener = opener

    def search(self, query: str, language_code: str, max_results: int = 10) -> list[dict[str, Any]]:
        params = {"part": "snippet", "type": "video", "q": query,
                  "relevanceLanguage": language_code, "maxResults": max_results,
                  "key": self.api_key,
                  "fields": "items(id/videoId,snippet(title,description,channelTitle,thumbnails))"}
        request = Request(self.endpoint + "?" + urlencode(params), headers={"Accept": "application/json"})
        try:
            with self.opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                detail = json.loads(exc.read().decode("utf-8"))
                reason = detail.get("error", {}).get("errors", [{}])[0].get("reason", "")
            except (ValueError, AttributeError):
                reason = ""
            if reason in {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"}:
                raise YouTubeDiscoveryError("YouTube search quota is exhausted. Try again later or use the search link.", "quota_exceeded") from exc
            raise YouTubeDiscoveryError("YouTube search is temporarily unavailable. Try again later.") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise YouTubeDiscoveryError("Could not reach YouTube search. Check your connection or use the search link.", "network_error") from exc
        except Exception as exc:
            if isinstance(exc, YouTubeDiscoveryError):
                raise
            raise YouTubeDiscoveryError("YouTube search returned an unreadable response.", "invalid_response") from exc
        output: list[dict[str, Any]] = []
        for result in payload.get("items", []):
            video_id = (result.get("id") or {}).get("videoId")
            snippet = result.get("snippet") or {}
            thumbnails = snippet.get("thumbnails") or {}
            thumbnail = next((thumbnails[key].get("url") for key in ("high", "medium", "default")
                              if isinstance(thumbnails.get(key), dict) and thumbnails[key].get("url")), None)
            if thumbnail:
                try:
                    parsed_thumbnail = urlsplit(thumbnail)
                    if parsed_thumbnail.scheme != "https" or not (
                        (parsed_thumbnail.hostname or "").lower().endswith("ytimg.com")
                        or (parsed_thumbnail.hostname or "").lower().endswith("ggpht.com")
                    ):
                        thumbnail = None
                except ValueError:
                    thumbnail = None
            output.append({"video_id": video_id, "title": snippet.get("title"),
                           "description": snippet.get("description"),
                           "channel_title": snippet.get("channelTitle"),
                           "thumbnail_url": thumbnail})
        return output
