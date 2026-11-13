"""Import and rank Tunely's manually reviewed song catalog."""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import sqlite3
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REVIEWED_LANGUAGES = ("Telugu", "Hindi", "Tamil", "Malayalam", "Kannada", "English")
APPROVED_GENRES = ("Sad", "Party", "Mass", "Romantic", "Old", "Devotional")
GENRE_BY_KEY = {label.casefold(): label for label in APPROVED_GENRES}
DEFAULT_REVIEWED_WORKBOOK = Path(__file__).resolve().parents[2] / "data" / "telugu_songs_reviewed.xlsx"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    value = " ".join(str(value).split())
    return value or None


def _identity_part(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(value.casefold().split())


def _genre_labels(raw: Any, year: int | None) -> tuple[list[str], int]:
    labels: list[str] = []
    for value in re.split(r"[;,|]", str(raw or "")):
        normalized = GENRE_BY_KEY.get(value.strip().casefold())
        if normalized and normalized not in labels:
            labels.append(normalized)
    old_adjustments = 0
    if year is not None and year <= 2009 and "Old" not in labels:
        labels.append("Old")
        old_adjustments += 1
    elif year is not None and year >= 2010 and "Old" in labels:
        labels.remove("Old")
        old_adjustments += 1
    # Keep a stable, approved order for cards and matching.
    return [label for label in APPROVED_GENRES if label in labels], old_adjustments


def ensure_reviewed_catalog_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS reviewed_songs (
            song_key TEXT PRIMARY KEY,
            itunes_track_id TEXT UNIQUE,
            identity_key TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            artist_name TEXT,
            album_name TEXT,
            language TEXT NOT NULL,
            language_original TEXT NOT NULL,
            release_date TEXT,
            release_year INTEGER,
            primary_genre_raw TEXT NOT NULL,
            genres_json TEXT NOT NULL,
            artwork_url TEXT,
            track_url TEXT,
            preview_url TEXT,
            duration_ms INTEGER,
            language_status TEXT,
            discovered_by TEXT,
            genre_classification_evidence TEXT,
            source_row INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            imported_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reviewed_songs_language ON reviewed_songs(language);
        CREATE INDEX IF NOT EXISTS idx_reviewed_songs_year ON reviewed_songs(release_year);
    """)


def _excel_date(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except (ValueError, AttributeError):
            pass
    text = _text(value)
    if not text:
        return None
    if re.fullmatch(r"\d{5}(?:\.0+)?", text):
        from datetime import date, timedelta
        try:
            return (date(1899, 12, 30) + timedelta(days=int(float(text)))).isoformat()
        except (ValueError, OverflowError):
            pass
    return text[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", text) else text


def _xlsx_sheets(path: Path) -> list[tuple[str, list[tuple[str, ...]]]]:
    """Read cached cell values from an .xlsx using only the standard library."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
          "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
          "pkg": "http://schemas.openxmlformats.org/package/2006/relationships"}
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships.findall("pkg:Relationship", ns)}
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.findall(".//main:t", ns))
                      for item in shared_root.findall("main:si", ns)]
        else:
            shared = []
        output = []
        for sheet in workbook.findall("main:sheets/main:sheet", ns):
            relation_id = sheet.attrib.get(f"{{{ns['rel']}}}id")
            target = targets.get(relation_id or "")
            if not target:
                continue
            sheet_path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
            root = ET.fromstring(archive.read(sheet_path))
            all_rows: list[tuple[str, ...]] = []
            for row in root.findall(".//main:sheetData/main:row", ns):
                values: dict[int, str] = {}
                max_column = -1
                for cell in row.findall("main:c", ns):
                    ref = cell.attrib.get("r", "")
                    letters = re.match(r"([A-Z]+)", ref)
                    if not letters:
                        continue
                    column = 0
                    for letter in letters.group(1):
                        column = column * 26 + ord(letter) - 64
                    column -= 1
                    max_column = max(max_column, column)
                    value_node = cell.find("main:v", ns)
                    cell_type = cell.attrib.get("t")
                    if cell_type == "inlineStr":
                        value = "".join(node.text or "" for node in cell.findall(".//main:t", ns))
                    elif value_node is None:
                        value = ""
                    elif cell_type == "s":
                        value = shared[int(value_node.text or "0")]
                    else:
                        value = value_node.text or ""
                    values[column] = value
                all_rows.append(tuple(values.get(index, "") for index in range(max_column + 1)))
            output.append((sheet.attrib.get("name", ""), all_rows))
        return output


def import_reviewed_workbook(conn: sqlite3.Connection, workbook_path: str | Path) -> dict[str, Any]:
    """Idempotently import the primary worksheet without modifying the workbook."""
    path = Path(workbook_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Reviewed music workbook was not found: {path}")
    sheet_rows = _xlsx_sheets(path)
    selected = next((rows for _, rows in sheet_rows if rows and any(str(value).strip() == "Song Title" for value in rows[0])), None)
    if selected is None:
        raise ValueError("Workbook has no worksheet with a Song Title header.")
    raw_headers = selected[0]
    headers = {str(value).strip(): index for index, value in enumerate(raw_headers) if value is not None}
    required = {"Song Title", "Artist Name", "Album Name", "Language", "Release Date", "Release Year",
                "iTunes Track ID", "Primary Genre", "Artwork URL", "Track URL", "Preview URL", "Duration (ms)"}
    missing = sorted(required - headers.keys())
    if missing:
        raise ValueError(f"Reviewed workbook is missing required columns: {', '.join(missing)}")
    all_rows = selected[1:]

    ensure_reviewed_catalog_schema(conn)
    imported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen_tracks: set[str] = set()
    seen_identity: set[str] = set()
    duplicates_removed = invalid_rows = missing_metadata = old_adjustments = 0
    inserted = updated = 0
    def value(row: tuple[Any, ...], name: str) -> Any:
        index = headers[name]
        return row[index] if index < len(row) else None

    for row_number, row in enumerate(all_rows, start=2):
        title = _text(value(row, "Song Title"))
        artist = _text(value(row, "Artist Name"))
        album = _text(value(row, "Album Name"))
        language_original = _text(value(row, "Language"))
        if not title or not language_original:
            invalid_rows += 1
            continue
        lang_key = language_original.casefold()
        language = next((item for item in REVIEWED_LANGUAGES if item.casefold() == lang_key), "Unknown")
        raw_id = _text(value(row, "iTunes Track ID"))
        track_id = raw_id if raw_id and raw_id.isdigit() else None
        identity_parts = (_identity_part(title), _identity_part(artist), _identity_part(album))
        identity_digest = hashlib.sha256("\x1f".join(identity_parts).encode("utf-8")).hexdigest()
        identity_key = f"itunes:{track_id}" if track_id else f"fallback:{identity_digest}"
        if (track_id and track_id in seen_tracks) or identity_key in seen_identity:
            duplicates_removed += 1
            continue
        seen_identity.add(identity_key)
        if track_id:
            seen_tracks.add(track_id)
        raw_genre = _text(value(row, "Primary Genre")) or ""
        raw_year = value(row, "Release Year")
        try:
            year = int(raw_year) if raw_year is not None and str(raw_year).strip() else None
        except (TypeError, ValueError):
            year = None
        genres, adjusted = _genre_labels(raw_genre, year)
        old_adjustments += adjusted
        duration_raw = value(row, "Duration (ms)")
        try:
            duration = int(duration_raw) if duration_raw is not None else None
        except (TypeError, ValueError):
            duration = None
        meta = [artist, album, language, _excel_date(value(row, "Release Date")), year, track_id,
                raw_genre, _text(value(row, "Artwork URL")), _text(value(row, "Track URL")),
                _text(value(row, "Preview URL")), duration]
        missing_metadata += sum(item is None for item in meta)
        key = f"itunes:{track_id}" if track_id else identity_key
        record = (key, track_id, identity_key, title, artist, album, language, language_original,
                  _excel_date(value(row, "Release Date")), year, raw_genre,
                  json.dumps(genres, ensure_ascii=False), _text(value(row, "Artwork URL")),
                  _text(value(row, "Track URL")), _text(value(row, "Preview URL")), duration,
                  _text(value(row, "Language Status")), _text(value(row, "Discovered By")),
                  _text(value(row, "Genre Classification Evidence")), row_number,
                  path.name, imported_at)
        exists = conn.execute("SELECT 1 FROM reviewed_songs WHERE song_key=? OR identity_key=?", (key, identity_key)).fetchone()
        conn.execute("""INSERT INTO reviewed_songs(song_key,itunes_track_id,identity_key,title,artist_name,album_name,
            language,language_original,release_date,release_year,primary_genre_raw,genres_json,artwork_url,track_url,
            preview_url,duration_ms,language_status,discovered_by,genre_classification_evidence,source_row,source_file,imported_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(song_key) DO UPDATE SET itunes_track_id=excluded.itunes_track_id,identity_key=excluded.identity_key,
            title=excluded.title,artist_name=excluded.artist_name,album_name=excluded.album_name,language=excluded.language,
            language_original=excluded.language_original,release_date=excluded.release_date,release_year=excluded.release_year,
            primary_genre_raw=excluded.primary_genre_raw,genres_json=excluded.genres_json,artwork_url=excluded.artwork_url,
            track_url=excluded.track_url,preview_url=excluded.preview_url,duration_ms=excluded.duration_ms,
            language_status=excluded.language_status,discovered_by=excluded.discovered_by,
            genre_classification_evidence=excluded.genre_classification_evidence,source_row=excluded.source_row,
            source_file=excluded.source_file,imported_at=excluded.imported_at""", record)
        if exists:
            updated += 1
        else:
            inserted += 1
    return {"rows_in_workbook": len(all_rows), "inserted": inserted, "updated": updated,
            "duplicates_removed": duplicates_removed, "invalid_rows": invalid_rows,
            "missing_metadata_values": missing_metadata, "old_label_adjustments": old_adjustments,
            "imported_count": conn.execute("SELECT COUNT(*) FROM reviewed_songs").fetchone()[0]}


def genre_targets(mood: str, goal: str, intensity: str) -> dict[str, float]:
    mood_targets: dict[str, list[str]] = {
        "sad": ["Sad"], "happy": ["Party"], "energetic": ["Party", "Mass"],
        "calm": ["Devotional", "Romantic", "Sad"], "stressed": ["Devotional", "Sad", "Romantic"],
        "romantic": ["Romantic"], "motivated": ["Mass", "Party"], "nostalgic": ["Old"], "neutral": [],
    }
    goal_targets = {
        "relax": ["Devotional", "Romantic", "Sad"], "wind down": ["Devotional", "Romantic", "Sad"],
        "focus": ["Devotional", "Sad"], "energize": ["Party", "Mass"],
        "feel-good": ["Party", "Mass"], "discover something new": list(APPROVED_GENRES),
    }
    targets: dict[str, float] = {}
    for rank, category in enumerate(mood_targets.get(mood, [])):
        targets[category] = targets.get(category, 0) + (4.0 if rank == 0 else 2.5 if rank == 1 else 1.5)
    for rank, category in enumerate(goal_targets.get(goal, [])):
        targets[category] = targets.get(category, 0) + (2.0 if rank == 0 else 1.5 if rank == 1 else 1.0)
    if mood == "neutral" and not targets:
        return {label: 1.0 for label in APPROVED_GENRES}
    if intensity == "intense":
        targets = {key: score * (1.35 if score >= 4 else 1.1) for key, score in targets.items()}
    elif intensity == "mild":
        targets = {key: score * (0.8 if score >= 4 else 1.0) for key, score in targets.items()}
    return targets


def _row_payload(row: sqlite3.Row, *, mood: str, intensity: str, goal: str | None,
                 targets: dict[str, float], genre_filter: str | None = None) -> dict[str, Any]:
    genres = json.loads(row["genres_json"] or "[]")
    matched = [genre for genre in genres if genre in targets]
    score = sum(targets.get(genre, 0) for genre in matched)
    category_text = ", ".join(genres) if genres else "no approved genre label"
    if genre_filter:
        reason = f"Matches the selected {genre_filter} genre from your reviewed catalog."
    elif matched:
        reason = f"Matches your {mood} mood through the reviewed {', '.join(matched)} classification"
        if goal:
            reason += f" and your {goal} listening goal"
        reason += "."
    elif mood == "neutral":
        reason = f"Selected for neutral discovery from the reviewed {category_text} catalog labels."
    else:
        reason = "Matches your selected language; no manually reviewed genre tag directly matches this mood."
    original_genres = [GENRE_BY_KEY.get(value.strip().casefold()) for value in re.split(r"[;,|]", row["primary_genre_raw"] or "")]
    original_had_old = "Old" in original_genres
    normalized_old = "Old" in genres
    if row["release_year"] is not None and original_had_old != normalized_old:
        category_explanation = (
            "Old tag normalized by the fixed release-year cutoff; original Primary Genre retained verbatim."
        )
    else:
        category_explanation = row["genre_classification_evidence"] or "Manually reviewed genre tags."
    return {
        "song_id": row["song_key"], "song_name": row["title"], "title": row["title"],
        "artist": row["artist_name"], "album_name": row["album_name"],
        "language": row["language"], "language_verified": row["language"] != "Unknown",
        "language_evidence": "Language column in manually reviewed dataset",
        "release_date": row["release_date"], "release_year": row["release_year"],
        "itunes_track_id": row["itunes_track_id"], "primary_genre_raw": row["primary_genre_raw"], "genre": category_text,
        "track_genre": category_text, "music_categories": genres,
        "category_explanation": category_explanation,
        "genre_source": "reviewed_dataset", "cover_image": row["artwork_url"],
        "artwork_url": row["artwork_url"], "track_url": row["track_url"],
        "preview_url": row["preview_url"], "duration_ms": row["duration_ms"],
        "youtube_music_search_url": _youtube_music_search(row["title"], row["artist_name"]),
        "selected_mood": mood if mood != "neutral" or not genre_filter else None,
        "selected_genre": genre_filter, "mood_intensity": intensity.title(),
        "selected_goal": goal, "explanation": reason, "score": score,
    }


def _youtube_music_search(title: str, artist: str | None) -> str:
    from urllib.parse import urlencode
    return "https://music.youtube.com/search?" + urlencode({"q": " ".join(part for part in (title, artist) if part)})


def rank_reviewed_songs(conn: sqlite3.Connection, *, user_id: str, language: str, mood: str,
                        intensity: str, goal: str = "", limit: int = 8,
                        exclude_track_ids: Iterable[str] = (), recent_days: int = 14,
                        genre_filter: str | None = None,
                        genre_targets_override: dict[str, float] | None = None) -> dict[str, Any]:
    targets = ({genre_filter: {"mild": 0.8, "moderate": 1.0, "intense": 1.35}.get(intensity, 1.0)}
               if genre_filter else genre_targets_override or genre_targets(mood, goal, intensity))
    display_targets = targets
    if genre_targets_override and not genre_filter:
        # Keep explanations focused on labels with meaningful model support. A
        # top-label floor prevents a low-confidence model from hiding all songs.
        display_targets = {label: score for label, score in targets.items() if score >= 0.35}
        if not display_targets and targets:
            top = max(targets, key=targets.get)
            display_targets = {top: targets[top]}
    rows = conn.execute("SELECT * FROM reviewed_songs WHERE language=? ORDER BY title COLLATE NOCASE,song_key",
                        (language,)).fetchall()
    excluded = {str(value) for value in exclude_track_ids if value}
    recent = {str(row[0]) for row in conn.execute("""SELECT DISTINCT track_id FROM itunes_recommendations
        WHERE user_id=? AND created_at>=datetime('now', ?)""", (user_id, f"-{max(1, recent_days)} days"))}
    candidates = []
    for row in rows:
        item = _row_payload(row, mood=mood, intensity=intensity, goal=goal or None,
                            targets=display_targets, genre_filter=genre_filter)
        if genre_targets_override and not genre_filter:
            item["score"] = sum(targets.get(label, 0.0) for label in item["music_categories"])
        if genre_filter and genre_filter not in item["music_categories"]:
            continue
        if mood != "neutral" and item["score"] <= 0:
            continue
        candidates.append((row, item))
    fresh = [(row, item) for row, item in candidates
             if str(row["itunes_track_id"] or "") not in excluded | recent]
    if len(fresh) >= limit:
        candidates = fresh
    else:
        # Use the remaining unseen catalog before allowing any recently shown IDs.
        candidates = fresh + [(row, item) for row, item in candidates
                              if str(row["itunes_track_id"] or "") not in excluded | {str(r["itunes_track_id"] or "") for r, _ in fresh}]
    candidates.sort(key=lambda pair: (-pair[1]["score"], pair[1]["song_name"].casefold(), pair[1]["song_id"]))
    chosen: list[tuple[sqlite3.Row, dict[str, Any]]] = []
    artist_counts: dict[str, int] = {}
    album_counts: dict[str, int] = {}
    remaining = list(candidates)
    while remaining and len(chosen) < limit:
        best_index = min(range(len(remaining)), key=lambda i: (
            -remaining[i][1]["score"] + artist_counts.get(_identity_part(remaining[i][0]["artist_name"]), 0) * 1.25
            + album_counts.get(_identity_part(remaining[i][0]["album_name"]), 0) * 0.35,
            remaining[i][1]["song_name"].casefold(), remaining[i][1]["song_id"]))
        row, item = remaining.pop(best_index)
        chosen.append((row, item))
        artist_key, album_key = _identity_part(row["artist_name"]), _identity_part(row["album_name"])
        artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
        album_counts[album_key] = album_counts.get(album_key, 0) + 1
    return {"recommendations": [item for _, item in chosen], "candidate_count": len(candidates),
            "language_catalog_count": len(rows), "genre_targets": display_targets,
            "genre_relevance_scores": targets,
            "fresh_count": len(fresh), "rotation_fallback": len(fresh) < limit}


def reviewed_catalog_statistics(conn: sqlite3.Connection) -> dict[str, Any]:
    total = int(conn.execute("SELECT COUNT(*) FROM reviewed_songs").fetchone()[0])
    language_counts = {row[0]: row[1] for row in conn.execute(
        "SELECT language,COUNT(*) FROM reviewed_songs GROUP BY language ORDER BY language")}
    for language in REVIEWED_LANGUAGES:
        language_counts.setdefault(language, 0)
    genre_counts = {genre: 0 for genre in APPROVED_GENRES}
    for row in conn.execute("SELECT genres_json FROM reviewed_songs"):
        for genre in json.loads(row[0] or "[]"):
            if genre in genre_counts:
                genre_counts[genre] += 1
    return {"song_count": total, "language_counts": language_counts, "genre_counts": genre_counts}
