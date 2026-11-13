"""Safe external listening links derived from verified catalog metadata."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlencode, urlsplit


_ALLOWED_YOUTUBE_HOSTS = {"music.youtube.com", "www.youtube.com", "youtube.com", "youtu.be"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def verified_youtube_music_url(value: object) -> str | None:
    """Return an HTTPS YouTube URL only when it is an allowed, complete URL.

    A stored link is considered a direct candidate because it came from the
    catalog; callers remain responsible for maintaining its verification.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    try:
        parsed = urlsplit(candidate)
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in _ALLOWED_YOUTUBE_HOSTS:
            return None
        if parsed.username or parsed.password:
            return None
        if parsed.hostname in {"music.youtube.com", "www.youtube.com", "youtube.com"} and parsed.path != "/watch":
            return None
        if parsed.hostname in {"music.youtube.com", "www.youtube.com", "youtube.com"} and not any(
            _VIDEO_ID.fullmatch(video_id) for video_id in parse_qs(parsed.query).get("v", [])
        ):
            return None
        if parsed.hostname == "youtu.be" and not _VIDEO_ID.fullmatch(parsed.path.strip("/")):
            return None
        return candidate
    except ValueError:
        return None


def youtube_music_search_url(title: object, artist: object) -> str | None:
    """Build a YouTube Music search URL without implying an exact-song match."""
    clean_title = title.strip() if isinstance(title, str) else ""
    clean_artist = artist.strip() if isinstance(artist, str) else ""
    query = " ".join(part for part in (clean_title, clean_artist) if part)
    if not query:
        return None
    return "https://music.youtube.com/search?" + urlencode({"q": query})


def listening_links(title: object, artist: object, stored_url: object = None) -> dict[str, str | None]:
    """Return a verified direct link when available and always a search fallback."""
    return {
        "youtube_music_url": verified_youtube_music_url(stored_url),
        "youtube_music_search_url": youtube_music_search_url(title, artist),
    }
