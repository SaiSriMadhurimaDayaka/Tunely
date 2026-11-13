"""Export locally stored and additionally discovered Telugu-targeted iTunes songs.

This standalone utility reads the local catalog but does not mutate it. Apple's
iTunes Search API does not document offset pagination, so it makes distinct
bounded searches and deduplicates their overlapping song IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
import unicodedata
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.recommender.itunes_discovery import ITunesDiscoveryError, ITunesSearchClient, _language_confidence
from backend.recommender.ranking import DEFAULT_DB
from backend.recommender.youtube_discovery import SUPPORTED_LANGUAGES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "telugu_songs.csv"
SEARCH_QUERIES = (
    "Telugu songs", "Telugu film songs", "Telugu movie songs", "Telugu soundtrack songs",
    "Tollywood songs", "Telugu old songs", "Telugu classic songs", "Telugu latest songs",
    "Telugu new songs", "Telugu love songs", "Telugu romantic songs", "Telugu sad songs",
    "Telugu folk songs", "Telugu melody songs", "Telugu mass songs", "Telugu devotional songs",
    "Telugu paatalu", "Telugu paatalu old", "తెలుగు పాటలు", "తెలుగు సినిమా పాటలు",
    "తెలుగు పాత పాటలు", "తెలుగు ప్రేమ పాటలు", "తెలుగు జానపద పాటలు",
)
CSV_FIELDS = (
    "Song Title", "Artist Name", "Album Name", "Language", "Release Date", "Release Year",
    "iTunes Track ID", "Primary Genre", "Artwork URL", "Track URL", "Preview URL",
    "Duration (ms)", "Language Status", "Discovered By",
)
METADATA_FIELDS = (
    "Song Title", "Artist Name", "Album Name", "Language", "Release Date", "Release Year",
    "iTunes Track ID", "Primary Genre", "Artwork URL", "Track URL", "Preview URL", "Duration (ms)",
)


def _clean_text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _normalized_key_part(value: Any) -> str:
    value = unicodedata.normalize("NFKC", _clean_text(value)).casefold()
    return " ".join(value.split())


def _dedupe_key(item: Mapping[str, Any]) -> tuple[str, ...]:
    track_id = item.get("trackId") or item.get("track_id") or item.get("iTunes Track ID")
    if track_id is not None and str(track_id).strip():
        return ("id", str(track_id).strip())
    title = item.get("trackName") or item.get("Song Title") or item.get("title")
    artist = item.get("artistName") or item.get("Artist Name") or item.get("artist")
    album = item.get("collectionName") or item.get("Album Name") or item.get("album_name")
    return ("metadata", _normalized_key_part(title), _normalized_key_part(artist), _normalized_key_part(album))


def _https(value: Any) -> str:
    url = _clean_text(value)
    return "https://" + url[7:] if url.startswith("http://") else url


def _release_date(item: Mapping[str, Any]) -> tuple[str, str]:
    raw = item.get("releaseDate") or item.get("Release Date") or item.get("release_date")
    if not isinstance(raw, str):
        return "", ""
    raw = raw.strip()
    try:
        parsed = date.fromisoformat(raw[:10])
        return raw, str(parsed.year)
    except ValueError:
        year = raw[:4] if len(raw) >= 4 and raw[:4].isdigit() else ""
        return raw, year


def _row(item: Mapping[str, Any], *, source: str) -> dict[str, str] | None:
    title = _clean_text(item.get("trackName") or item.get("Song Title") or item.get("title"))
    if not title:
        return None
    artist = _clean_text(item.get("artistName") or item.get("Artist Name") or item.get("artist"))
    album = _clean_text(item.get("collectionName") or item.get("Album Name") or item.get("album_name"))
    genre = _clean_text(item.get("primaryGenreName") or item.get("Primary Genre"))
    evidence_text = " ".join((title, artist, album, genre))
    confidence, evidence = _language_confidence(evidence_text, "Telugu")
    script_evidence = bool(re.search(SUPPORTED_LANGUAGES["Telugu"]["script"], evidence_text))
    stored_language = item.get("_tunely_language")
    if stored_language == "Telugu":
        language = "Telugu"
        language_status = f"stored database estimate: {item.get('_tunely_language_evidence') or 'unspecified evidence'}"
    elif confidence >= 0.90 and script_evidence:
        language = "Telugu"
        language_status = "inferred from Telugu script in catalog text"
    else:
        language = "Unknown"
        language_status = "unverified in available song metadata"
    release, year = _release_date(item)
    duration = item.get("trackTimeMillis") or item.get("Duration (ms)") or item.get("duration_ms")
    duration_text = str(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0 else ""
    return {
        "Song Title": title,
        "Artist Name": artist,
        "Album Name": album,
        "Language": language,
        "Release Date": release,
        "Release Year": year,
        "iTunes Track ID": str(item.get("trackId") or item.get("track_id") or item.get("iTunes Track ID") or ""),
        "Primary Genre": genre,
        "Artwork URL": _https(item.get("artworkUrl100") or item.get("artworkUrl60") or item.get("Artwork URL")),
        "Track URL": _https(item.get("trackViewUrl") or item.get("Track URL")),
        "Preview URL": _https(item.get("previewUrl") or item.get("Preview URL")),
        "Duration (ms)": duration_text,
        "Language Status": language_status,
        "Discovered By": source,
    }


def _local_records(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return unique Telugu-targeted and reliably detected Telugu local records."""
    records: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT DISTINCT t.raw_json,t.language,t.language_evidence FROM itunes_tracks t LEFT JOIN itunes_track_languages l USING(track_id) "
        "WHERE l.requested_language='Telugu' OR t.language='Telugu'"
    ):
        try:
            payload = json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("kind") == "song":
            payload["_tunely_language"] = row[1]
            payload["_tunely_language_evidence"] = row[2]
            records.append(payload)
    return records


def export_telugu_songs(*, database: str | Path = DEFAULT_DB,
                        output: str | Path = DEFAULT_OUTPUT,
                        country: str = "in", limit: int = 200,
                        delay_seconds: float = 1.1,
                        queries: Sequence[str] = SEARCH_QUERIES,
                        client: ITunesSearchClient | None = None) -> dict[str, Any]:
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    if delay_seconds < 0:
        raise ValueError("delay cannot be negative")
    database, output = Path(database), Path(output)
    if not database.exists():
        raise FileNotFoundError(f"Local Tunely database not found: {database}")
    api = client or ITunesSearchClient(
        timeout=max(2.0, min(30.0, float(os.environ.get("ITUNES_SEARCH_TIMEOUT_SECONDS", "8"))))
    )
    all_items: dict[tuple[str, ...], dict[str, Any]] = {}
    local_duplicates = 0
    with closing(sqlite3.connect(database, timeout=20)) as conn:
        local = _local_records(conn)
        for item in local:
            key = _dedupe_key(item)
            row = _row(item, source="Local iTunes dataset")
            if not row:
                continue
            if key in all_items:
                local_duplicates += 1
                continue
            all_items[key] = row
    local_count = len(all_items)
    attempted = successful = raw_results = api_duplicates = failed = 0
    rate_limited = False
    errors: list[dict[str, str]] = []
    previous_request = 0.0
    for query in dict.fromkeys(q.strip() for q in queries if q and q.strip()):
        remaining = delay_seconds - (time.monotonic() - previous_request)
        if previous_request and remaining > 0:
            time.sleep(remaining)
        attempted += 1
        try:
            records = api.search(query, country, limit)
            previous_request = time.monotonic()
            successful += 1
            raw_results += len(records)
        except ITunesDiscoveryError as exc:
            failed += 1
            errors.append({"query": query, "code": exc.code, "message": str(exc)})
            print(f"iTunes query failed ({exc.code}): {query}", file=sys.stderr, flush=True)
            if exc.code == "rate_limited":
                rate_limited = True
                break
            continue
        for item in records:
            key = _dedupe_key(item)
            row = _row(item, source=f"iTunes Search API: {query}")
            if not row:
                continue
            if key in all_items:
                api_duplicates += 1
                current = all_items[key]
                # Keep metadata already present and fill only blanks from another
                # genuine result for that same provider ID or identity tuple.
                for field in CSV_FIELDS:
                    if not current.get(field) and row.get(field):
                        current[field] = row[field]
                if query not in current["Discovered By"]:
                    current["Discovered By"] += f"; {query}"
            else:
                all_items[key] = row

    rows = list(all_items.values())
    rows.sort(key=lambda item: (item["Song Title"].casefold(), item["Artist Name"].casefold(), item["iTunes Track ID"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    def missing(row: Mapping[str, str], field: str) -> bool:
        if field == "Language":
            return row.get(field) != "Telugu"
        return not row.get(field, "").strip()

    missing_by_field = {field: sum(missing(row, field) for row in rows) for field in METADATA_FIELDS}
    summary = {
        "output_csv": str(output.resolve()),
        "unique_songs": len(rows),
        "songs_from_existing_database": local_count,
        "additional_songs_from_itunes": len(rows) - local_count,
        "duplicate_records_removed": local_duplicates + api_duplicates,
        "songs_with_missing_metadata": sum(any(missing(row, field) for field in METADATA_FIELDS) for row in rows),
        "missing_metadata_by_field": missing_by_field,
        "language_marked_telugu": sum(row["Language"] == "Telugu" for row in rows),
        "language_from_stored_database_estimate": sum(row["Language Status"].startswith("stored database estimate") for row in rows),
        "language_unverified": sum(row["Language"] == "Unknown" for row in rows),
        "api": {"queries_attempted": attempted, "queries_succeeded": successful,
                "raw_song_results": raw_results, "limit_per_query": limit,
                "failed_queries": failed, "rate_limited": rate_limited, "errors": errors},
        "limitations": [
            "The CSV includes all existing Telugu-targeted local records plus unique results from the listed targeted searches.",
            "iTunes Search API documentation does not specify offset pagination for this endpoint; distinct queries expand coverage but cannot guarantee the entire catalog.",
            "Language is marked Telugu when native Telugu script appears in catalog text or the existing database stored a Telugu estimate; that estimate is labeled as such, and remaining targeted results are marked Unknown.",
        ],
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Telugu-targeted iTunes songs and enrich the export with distinct searches.")
    parser.add_argument("--database", type=Path, default=Path(os.environ.get("TUNELY_DATABASE", DEFAULT_DB)))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--country", default=os.environ.get("ITUNES_SEARCH_COUNTRY", "in"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--delay", type=float, default=1.1)
    args = parser.parse_args()
    try:
        summary = export_telugu_songs(database=args.database, output=args.output,
                                      country=args.country, limit=args.limit, delay_seconds=args.delay)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if summary["api"]["failed_queries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
