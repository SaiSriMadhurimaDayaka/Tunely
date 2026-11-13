"""LRCLIB client with strict title/artist validation and no lyrics persistence."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class LyricsProviderError(RuntimeError):
    def __init__(self, message: str, code: str = "lyrics_provider_error", retry_after: float | None = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


def _fold(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold()
    text = re.sub(r"\b(feat(?:uring)?|ft)\b.*$", "", text)
    text = re.sub(r"\s*[\[(](?:live|radio edit|remaster(?:ed)?|deluxe edition|version)[\])]\s*$", "", text)
    text = " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))
    return text.strip()


def _similarity(left: Any, right: Any) -> float:
    a, b = _fold(left), _fold(right)
    if not a or not b:
        return 0.0
    if a == b or a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a, b).ratio()


def select_verified_lyrics(records: Any, track: Mapping[str, Any]) -> dict[str, Any] | None:
    """Choose a close title AND artist match; duration can reject wrong versions."""
    if not isinstance(records, list):
        return None
    wanted_title = track.get("trackName") or track.get("title") or track.get("song_name")
    wanted_artist = track.get("artistName") or track.get("artist")
    wanted_album = track.get("collectionName") or track.get("album_name")
    duration_ms = track.get("trackTimeMillis") or track.get("duration_ms")
    try:
        wanted_duration = float(duration_ms) / 1000 if duration_ms else None
    except (TypeError, ValueError):
        wanted_duration = None
    best: tuple[float, dict[str, Any]] | None = None
    for record in records:
        if not isinstance(record, dict):
            continue
        title_score = _similarity(wanted_title, record.get("trackName") or record.get("name"))
        artist_score = _similarity(wanted_artist, record.get("artistName"))
        if title_score < 0.84 or artist_score < 0.78:
            continue
        try:
            result_duration = float(record.get("duration")) if record.get("duration") else None
        except (TypeError, ValueError):
            result_duration = None
        if wanted_duration and result_duration and abs(wanted_duration - result_duration) > 12:
            continue
        album_score = _similarity(wanted_album, record.get("albumName")) if wanted_album else 0
        score = 0.55 * title_score + 0.40 * artist_score + 0.05 * album_score
        if best is None or score > best[0]:
            best = (score, record)
    if best is None:
        return None
    return {**best[1], "_match_score": round(best[0], 4)}


class LRCLibClient:
    endpoint = "https://lrclib.net/api/search"

    def __init__(self, *, timeout: float = 4.0, opener=urlopen):
        self.timeout = timeout
        self.opener = opener
        self._blocked_until = 0.0

    def lookup(self, track: Mapping[str, Any]) -> dict[str, Any]:
        title = track.get("trackName") or track.get("title") or track.get("song_name")
        artist = track.get("artistName") or track.get("artist")
        if not isinstance(title, str) or not title.strip() or not isinstance(artist, str) or not artist.strip():
            return {"status": "missing_identity", "lyrics_available": False}
        if time.monotonic() < self._blocked_until:
            remaining = self._blocked_until - time.monotonic()
            raise LyricsProviderError(f"LRCLIB asked Tunely to wait {remaining:.0f} seconds before another lookup.",
                                      "lyrics_rate_limited", remaining)
        params = urlencode({"track_name": title.strip(), "artist_name": artist.strip()})
        request = Request(f"{self.endpoint}?{params}", headers={
            "Accept": "application/json", "User-Agent": "Tunely/0.1 (https://github.com/Tunely)"})
        try:
            with self.opener(request, timeout=self.timeout) as response:
                body = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return {"status": "not_found", "lyrics_available": False}
            if exc.code in (429, 503):
                try:
                    default_wait = "60" if exc.code == 429 else "1"
                    retry_after = max(1.0, min(3600.0, float(exc.headers.get("Retry-After", default_wait))))
                except (AttributeError, TypeError, ValueError):
                    retry_after = 60.0 if exc.code == 429 else 1.0
                self._blocked_until = time.monotonic() + retry_after
                exc.close()
                code = "lyrics_rate_limited" if exc.code == 429 else "lyrics_server_busy"
                message = "rate limited" if exc.code == 429 else "temporarily unavailable"
                raise LyricsProviderError(f"LRCLIB is {message}; retry after about {retry_after:.0f} seconds.",
                                          code, retry_after) from exc
            raise LyricsProviderError(f"LRCLIB returned HTTP {exc.code}.", "lyrics_http_error") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LyricsProviderError(f"Could not reach LRCLIB ({exc}).", "lyrics_network_error") from exc
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise LyricsProviderError("LRCLIB returned invalid JSON.", "lyrics_invalid_response") from exc
        if not isinstance(payload, list):
            raise LyricsProviderError("LRCLIB response did not contain a result list.", "lyrics_invalid_response")
        matched = select_verified_lyrics(payload, track)
        if not matched:
            return {"status": "unmatched", "lyrics_available": False}
        lyric_text = matched.get("plainLyrics") or matched.get("syncedLyrics")
        if not isinstance(lyric_text, str) or not lyric_text.strip():
            return {"status": "instrumental" if matched.get("instrumental") else "no_lyrics_text",
                    "lyrics_available": False, "record_id": matched.get("id"),
                    "matched_title": matched.get("trackName") or matched.get("name"),
                    "matched_artist": matched.get("artistName"), "match_score": matched["_match_score"]}
        return {"status": "matched", "lyrics_available": True, "record_id": matched.get("id"),
                "matched_title": matched.get("trackName") or matched.get("name"),
                "matched_artist": matched.get("artistName"), "match_score": matched["_match_score"],
                "lyrics_text": lyric_text}
