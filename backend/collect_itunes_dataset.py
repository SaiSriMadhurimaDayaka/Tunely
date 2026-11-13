"""Collect/update the local iTunes song metadata catalog without an API key.

Run from the project root with ``python -m backend.collect_itunes_dataset``.
The public Search API has no documented offset pagination parameter, so this
collector uses distinct, language-targeted search terms with a bounded limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

from backend.recommender.itunes_dataset import dataset_statistics, ensure_itunes_dataset_schema, upsert_itunes_tracks
from backend.recommender.itunes_discovery import ITunesDiscoveryError, ITunesSearchClient
from backend.recommender.ranking import DEFAULT_DB


LANGUAGES = ("Telugu", "Hindi", "Tamil", "Malayalam", "Kannada", "English", "Punjabi", "Bengali")
SEARCH_SUFFIXES = (
    "songs", "film songs", "love songs", "dance songs", "classic songs", "new songs", "sad songs",
)


def collection_queries(language: str) -> list[str]:
    return list(dict.fromkeys(f"{language} {suffix}" for suffix in SEARCH_SUFFIXES))


def collect_dataset(*, database: str | Path = DEFAULT_DB, languages: Sequence[str] = LANGUAGES,
                    country: str = "in", limit: int = 200, delay_seconds: float = 1.1,
                    refresh_after_days: int = 30, force: bool = False,
                    client: ITunesSearchClient | None = None) -> dict:
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    if delay_seconds < 0 or refresh_after_days < 0:
        raise ValueError("delay and refresh interval cannot be negative")
    canonical_languages = []
    for value in languages:
        canonical = next((name for name in LANGUAGES if name.casefold() == value.strip().casefold()), None)
        if not canonical:
            raise ValueError(f"Unsupported collection language: {value}")
        if canonical not in canonical_languages:
            canonical_languages.append(canonical)
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    collector = client or ITunesSearchClient(
        timeout=max(2.0, min(30.0, float(os.environ.get("ITUNES_SEARCH_TIMEOUT_SECONDS", "8"))))
    )
    totals = {"inserted": 0, "updated": 0, "invalid": 0, "requests": 0,
              "skipped_queries": 0, "failed_queries": 0, "rate_limited": False}
    cutoff = (datetime.now(timezone.utc) - timedelta(days=refresh_after_days)).isoformat(timespec="seconds")
    previous_request = 0.0
    with closing(sqlite3.connect(database, timeout=20)) as conn:
        conn.row_factory = sqlite3.Row
        ensure_itunes_dataset_schema(conn)
        for language in canonical_languages:
            for query in collection_queries(language):
                query_key = hashlib.sha256(f"{query.casefold()}|{language.casefold()}|{country.casefold()}".encode()).hexdigest()
                previous = conn.execute("SELECT last_collected_at FROM itunes_collection_queries WHERE query_key=?",
                                        (query_key,)).fetchone()
                if not force and previous and previous["last_collected_at"] >= cutoff:
                    totals["skipped_queries"] += 1
                    continue
                remaining = delay_seconds - (time.monotonic() - previous_request)
                if previous_request and remaining > 0:
                    time.sleep(remaining)
                try:
                    records = collector.search(query, country, limit)
                    previous_request = time.monotonic()
                    totals["requests"] += 1
                except ITunesDiscoveryError as exc:
                    totals["failed_queries"] += 1
                    print(f"{language}: search failed ({exc.code}): {exc}", file=sys.stderr, flush=True)
                    if exc.code == "rate_limited":
                        totals["rate_limited"] = True
                        break
                    continue
                counts = upsert_itunes_tracks(conn, records, requested_language=language, country=country)
                totals["inserted"] += counts["inserted"]
                totals["updated"] += counts["updated"]
                totals["invalid"] += counts["invalid"]
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                conn.execute("INSERT INTO itunes_collection_queries(query_key,query_text,requested_language,country,result_count,last_collected_at) "
                             "VALUES(?,?,?,?,?,?) ON CONFLICT(query_key) DO UPDATE SET result_count=excluded.result_count,last_collected_at=excluded.last_collected_at",
                             (query_key, query, language, country, len(records), now))
                conn.commit()
                print(f"{language}: {query!r} returned {len(records)} tracks; added {counts['inserted']}, updated {counts['updated']}", flush=True)
            if totals["rate_limited"]:
                break
        totals["dataset"] = dataset_statistics(conn)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally collect real iTunes song metadata for Tunely.")
    parser.add_argument("--database", type=Path, default=Path(os.environ.get("TUNELY_DATABASE", DEFAULT_DB)))
    parser.add_argument("--languages", default="all", help="all or a comma-separated subset of the eight supported languages")
    parser.add_argument("--country", default=os.environ.get("ITUNES_SEARCH_COUNTRY", "in"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--delay", type=float, default=1.1, help="minimum seconds between API requests")
    parser.add_argument("--refresh-after-days", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="refresh previously collected query terms")
    args = parser.parse_args()
    languages = LANGUAGES if args.languages.strip().casefold() == "all" else tuple(
        value.strip() for value in args.languages.split(",") if value.strip()
    )
    try:
        result = collect_dataset(database=args.database, languages=languages, country=args.country,
                                 limit=args.limit, delay_seconds=args.delay,
                                 refresh_after_days=args.refresh_after_days, force=args.force)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["failed_queries"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
