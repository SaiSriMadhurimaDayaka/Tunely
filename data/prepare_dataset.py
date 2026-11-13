"""Clean the supplied Spotify catalog without modifying its source file.

The bundled spotify.zip contains dataset.csv, a 114k-row Spotify tracks catalog.
It has real audio features and duration_ms, but no release date or cover image.
This script retains those fields as unavailable rather than inferring them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "original" / "spotifyIndian.zip"
OUTPUT_CSV = ROOT / "data" / "spotify_cleaned.csv"
OUTPUT_DB = ROOT / "data" / "tunely.sqlite3"
FEATURES = (
    "danceability", "energy", "acousticness", "instrumentalness", "valence",
    "speechiness", "liveness", "tempo", "loudness",
)
REQUIRED = {"song_name", "artist", "popularity", "duration_ms"}


def normalize_header(header: str) -> str:
    return "_".join(header.strip().lower().replace("-", " ").split())


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def number(value: str, *, integer: bool = False) -> int | float | None:
    if not value:
        return None
    try:
        result = float(value)
        if not math.isfinite(result):
            return None
        return int(result) if integer and result.is_integer() else (None if integer else result)
    except (ValueError, TypeError):
        return None


def source_rows(source: Path) -> tuple[TextIO, str]:
    """Return a readable stream and its owner (kept separately for closing)."""
    if source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source)
        candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not candidates:
            archive.close()
            raise ValueError("The ZIP archive does not contain a CSV file.")
        member = next((name for name in candidates if Path(name).name.lower() == "dataset.csv"), candidates[0])
        binary = archive.open(member)
        import io

        return io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""), member
    return source.open("r", encoding="utf-8-sig", newline=""), source.name


def clean_record(raw: dict[str, str], source_fields: set[str], dataset_source: str = "") -> dict[str, object] | None:
    row = {normalize_header(k): (v.strip() if v else "") for k, v in raw.items() if k is not None}
    song = first(row, "song_name", "track_name", "name")
    artist = first(row, "artist", "artists", "artist_name")
    popularity = number(first(row, "popularity"), integer=True)
    duration = number(first(row, "duration_ms", "duration"), integer=True)
    if not song or not artist or popularity is None or duration is None or duration <= 0:
        return None
    if not 0 <= popularity <= 100:
        return None

    release_date = first(row, "release_date", "released_dates", "release") or None
    parsed_date = None
    if release_date:
        for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                parsed_date = datetime.strptime(release_date, fmt).date().isoformat()
                break
            except ValueError:
                pass
    track_id = first(row, "track_id", "id") or None
    stable_id = track_id or "song_" + hashlib.sha256(
        f"{song.casefold()}\0{artist.casefold()}".encode("utf-8")
    ).hexdigest()[:24]
    result: dict[str, object] = {
        "song_id": stable_id,
        "track_id": track_id,
        "song_name": song,
        "artist": artist,
        "album_name": first(row, "album_name", "album") or None,
        "release_date": parsed_date,
        "release_year": int(parsed_date[:4]) if parsed_date else None,
        "popularity": popularity,
        "duration_ms": duration,
        "duration_minutes": round(duration / 60_000, 6),
        "album_type": first(row, "album_type", "type") or None,
        "cover_image": first(row, "cover_image", "image_url", "cover_url") or None,
        "explicit": first(row, "explicit").lower() in {"true", "1", "yes"} if "explicit" in row else None,
        "track_genre": first(row, "track_genre", "genre") or None,
        "language": first(row, "language", "lang") or None,
        "mood": first(row, "mood", "mood_tag") or None,
        "youtube_music_url": first(row, "youtube_music_url") or None,
        "dataset_source": dataset_source or None,
        "dataset_license": None,
        "available_features": ",".join(feature for feature in FEATURES if number(first(row, feature)) is not None),
    }
    for feature in FEATURES:
        value = number(first(row, feature))
        result[feature] = value
    # Audio feature values must remain missing if the source did not provide them.
    return result


def schema_sql() -> str:
    feature_sql = ",\n".join(f"    {name} REAL" for name in FEATURES)
    return f"""CREATE TABLE IF NOT EXISTS songs (
    song_id TEXT PRIMARY KEY,
    track_id TEXT UNIQUE,
    song_name TEXT NOT NULL,
    artist TEXT NOT NULL,
    album_name TEXT,
    release_date TEXT,
    release_year INTEGER,
    popularity INTEGER NOT NULL CHECK(popularity BETWEEN 0 AND 100),
    duration_ms INTEGER NOT NULL CHECK(duration_ms > 0),
    duration_minutes REAL NOT NULL,
    album_type TEXT,
    cover_image TEXT,
    explicit INTEGER,
    track_genre TEXT,
    language TEXT,
    mood TEXT,
    youtube_music_url TEXT,
    dataset_source TEXT,
    dataset_license TEXT,
    available_features TEXT NOT NULL DEFAULT '',
{feature_sql}
);
CREATE INDEX IF NOT EXISTS idx_songs_popularity ON songs(popularity DESC);
CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs(track_genre);
"""


def prepare(source: Path, output: Path, database: Path | None = None, *, merge: bool = False) -> dict[str, object]:
    logging.info("Reading original catalog: %s", source)
    stream, member = source_rows(source)
    try:
        reader = csv.DictReader(stream)
        fields = {normalize_header(value) for value in (reader.fieldnames or [])}
        # Different source exports use different labels; required data is validated per row.
        if not fields:
            raise ValueError("The source CSV has no header row.")
        counts = {"read": 0, "kept": 0, "invalid_removed": 0, "duplicates_removed": 0}
        seen: set[tuple[str, str]] = set()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as target:
            columns = ["song_id", "track_id", "song_name", "artist", "album_name", "release_date", "release_year",
                       "popularity", "duration_ms", "duration_minutes", "album_type", "cover_image", "explicit",
                       "track_genre", "language", "mood", "youtube_music_url", "dataset_source", "dataset_license", "available_features", *FEATURES]
            writer = csv.DictWriter(target, fieldnames=columns)
            writer.writeheader()
            if database:
                database.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(database) as conn:
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.executescript(schema_sql())
                    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(songs)")}
                    for column in ("language", "mood", "youtube_music_url", "dataset_source", "dataset_license"):
                        if column not in existing_columns:
                            conn.execute(f"ALTER TABLE songs ADD COLUMN {column} TEXT")
                    if not merge:
                        conn.execute("DELETE FROM songs")
                    insert_sql = f"INSERT INTO songs ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})"
                    if merge:
                        updates = ",".join(f"{column}=COALESCE(excluded.{column},songs.{column})" for column in columns if column != "song_id")
                        insert_sql += f" ON CONFLICT(song_id) DO UPDATE SET {updates}"
                    batch: list[tuple[object, ...]] = []
                    for raw in reader:
                        counts["read"] += 1
                        item = clean_record(raw, fields, f"{source.name}:{member}")
                        if item is None:
                            counts["invalid_removed"] += 1
                            continue
                        key = (str(item["song_name"]).casefold(), str(item["artist"]).casefold())
                        if key in seen:
                            counts["duplicates_removed"] += 1
                            continue
                        seen.add(key)
                        writer.writerow(item)
                        batch.append(tuple(item.get(column) for column in columns))
                        counts["kept"] += 1
                        if len(batch) >= 1000:
                            conn.executemany(insert_sql, batch)
                            batch.clear()
                    if batch:
                        conn.executemany(insert_sql, batch)
            else:
                for raw in reader:
                    counts["read"] += 1
                    item = clean_record(raw, fields, f"{source.name}:{member}")
                    if item is None:
                        counts["invalid_removed"] += 1
                        continue
                    key = (str(item["song_name"]).casefold(), str(item["artist"]).casefold())
                    if key in seen:
                        counts["duplicates_removed"] += 1
                        continue
                    seen.add(key)
                    writer.writerow(item)
                    counts["kept"] += 1
    finally:
        stream.close()
    report = {
        "source": str(source), "source_member": member, "source_columns": sorted(fields),
        "audio_features_present_in_source": [name for name in FEATURES if name in fields],
        "audio_features_missing_from_source": [name for name in FEATURES if name not in fields],
        "release_date_present_in_source": bool(fields & {"release_date", "released_dates", "release"}),
        "cover_image_present_in_source": bool(fields & {"cover_image", "image_url", "cover_url"}),
        "duration_unit": (
            "milliseconds (explicit duration_ms source column)" if "duration_ms" in fields else
            "milliseconds (verified against the matching prior cleaned sample: Duration_Minutes = Duration / 60,000)"
            if "duration" in fields and "duration_minutes" not in fields else
            "milliseconds (Duration_Minutes cross-check available)" if "duration_minutes" in fields else
            "milliseconds (Spotify duration convention; confirm source provenance)"
        ),
        **counts,
        "output_csv": str(output), "output_database": str(database) if database else None,
    }
    report_path = output.with_name("preparation_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Preparation summary:\n%s", json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Original CSV or ZIP archive")
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV, help="Path for cleaned CSV")
    parser.add_argument("--database", type=Path, default=OUTPUT_DB, help="SQLite database to populate; use --no-database to skip")
    parser.add_argument("--no-database", action="store_true", help="Only write cleaned CSV and report")
    parser.add_argument("--merge", action="store_true", help="Merge by song_id while preserving existing songs and linked history")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if not args.source.is_file():
        parser.error(f"Source file does not exist: {args.source}")
    prepare(args.source, args.output, None if args.no_database else args.database, merge=args.merge)


if __name__ == "__main__":
    main()
