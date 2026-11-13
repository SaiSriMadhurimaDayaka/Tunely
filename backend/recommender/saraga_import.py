"""Import a configured Saraga metadata sample into Tunely's existing SQLite DB.

This script consumes a local MTG/saraga checkout. It never downloads data and
does not overwrite the existing Spotify ``songs`` table. Saraga audio files
are registered only when they already exist in the checkout.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any

from .ranking import DEFAULT_DB
from .saraga_ranking import ensure_schema


DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "data" / "original" / "saraga"
DEFAULT_IDS = Path(__file__).resolve().parents[2] / "data" / "saraga_sample_ids.csv"
SARAGA_LICENSE = "CC BY-NC 4.0 (audio and annotations); editorial metadata license not specified in repository LICENSE.md"
SARAGA_ATTRIBUTION = "Saraga collections, Music Technology Group, Universitat Pompeu Fabra; editorial metadata references MusicBrainz"
SARAGA_SOURCE = "MTG Saraga"


def _names(value: Any, *keys: str) -> list[str]:
    """Read names from Saraga's actual metadata lists without inventing values."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        value = [value]
    output: list[str] = []
    for item in value:
        candidate: Any = item
        if isinstance(item, dict):
            candidate = None
            for key in keys:
                if key in item:
                    candidate = item[key]
                    break
            if candidate is None and "artist" in item and isinstance(item["artist"], dict):
                candidate = item["artist"].get("name")
        if isinstance(candidate, str) and candidate.strip() and candidate.strip() not in output:
            output.append(candidate.strip())
    return output


def _concept_names(value: Any) -> list[str]:
    """Preserve Saraga's canonical and common concept labels for search."""
    if not isinstance(value, list):
        value = [value] if value is not None else []
    labels: list[str] = []
    for item in value:
        if isinstance(item, dict):
            for key in ("name", "common_name", "title"):
                candidate = item.get(key)
                if isinstance(candidate, str) and candidate.strip() and candidate.strip() not in labels:
                    labels.append(candidate.strip())
        elif isinstance(item, str) and item.strip() and item.strip() not in labels:
            labels.append(item.strip())
    return labels


def _relative_recording_path(source: Path, tradition: str, raw_path: str) -> Path:
    parts = PurePosixPath(raw_path.replace("\\", "/")).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Saraga filepath contains an unsafe path component.")
    return source / "dataset" / tradition / Path(*parts)


def _read_tempo(recording_path: Path, tradition: str) -> tuple[float | None, str | None]:
    """Read tempo annotations in their documented units and preserve raw rows."""
    annotation = Path(str(recording_path) + ".tempo-manual.txt")
    if not annotation.is_file():
        return None, None
    parsed: list[dict[str, float]] = []
    try:
        lines = annotation.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None, None
    tempo_bpm: float | None = None
    for line in lines:
        values: list[float] = []
        for token in line.strip().split(","):
            try:
                values.append(float(token.strip()))
            except ValueError:
                values = []
                break
        if not values:
            continue
        if tradition == "carnatic" and len(values) >= 2:
            row = {"tempo_apm": values[0], "tempo_bpm": values[1]}
            if values[1] > 0 and tempo_bpm is None:
                tempo_bpm = values[1]
        elif tradition == "hindustani" and len(values) >= 6:
            row = {
                "tempo_matras_per_minute": values[0], "matra_interval_seconds": values[1],
                "sama_interval_seconds": values[2], "matras_per_cycle": values[3],
                "start_seconds": values[4], "end_seconds": values[5],
            }
        else:
            continue
        parsed.append(row)
    return tempo_bpm, json.dumps(parsed, ensure_ascii=False) if parsed else None


def _metadata_row(source: Path, tradition: str, mapping: dict[str, str]) -> dict[str, Any] | None:
    recording_path = _relative_recording_path(source, tradition, mapping.get("filepath", ""))
    metadata_path = Path(str(recording_path) + ".json")
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None

    recording_id = str(metadata.get("mbid") or mapping.get("mbid") or "").strip()
    if not recording_id:
        return None
    title = str(metadata.get("title") or recording_path.name).strip()
    artists = metadata.get("artists") or []
    artist_names: list[str] = []
    for entry in artists:
        if isinstance(entry, dict) and isinstance(entry.get("artist"), dict):
            name = entry["artist"].get("name")
            if isinstance(name, str) and name.strip() and name.strip() not in artist_names:
                artist_names.append(name.strip())
    if not artist_names:
        artist_names = _names(metadata.get("album_artists"), "name")

    release_values = _names(metadata.get("release"), "title", "name") or _names(metadata.get("concert"), "title", "name")
    raga = _concept_names(metadata.get("raaga") or metadata.get("raags"))
    tala = _concept_names(metadata.get("taala") or metadata.get("taals"))
    form = _names(metadata.get("form"), "name", "title")
    laya = _names(metadata.get("laya"), "name", "title")
    work = _names(metadata.get("work"), "title", "name")

    length = metadata.get("length")
    try:
        duration_ms = int(length) if length is not None and int(length) > 0 else None
    except (TypeError, ValueError, OverflowError):
        duration_ms = None

    audio_candidates = (Path(str(recording_path) + ".mp3.mp3"), Path(str(recording_path) + ".mp3"))
    audio_path = next((candidate for candidate in audio_candidates if candidate.is_file()), None)
    try:
        stored_audio_path = audio_path.relative_to(source).as_posix() if audio_path else None
    except ValueError:
        stored_audio_path = None
        audio_path = None
    tempo_bpm, tempo_annotations = _read_tempo(recording_path, tradition)
    return {
        "recording_id": recording_id,
        "title": title or recording_id,
        "artist": ", ".join(artist_names) or None,
        "tradition": tradition,
        "release_info": "; ".join(release_values) or None,
        "raga": "; ".join(raga) or None,
        "tala": "; ".join(tala) or None,
        "laya": "; ".join(laya) or None,
        "form": "; ".join(form) or None,
        "work": "; ".join(work) or None,
        "tempo_bpm": tempo_bpm,
        "tempo_annotations_json": tempo_annotations,
        "duration_ms": duration_ms,
        "language": None,
        "audio_path": stored_audio_path,
        "audio_status": "available_local" if audio_path else "not_downloaded",
        "dataset_source": SARAGA_SOURCE,
        "dataset_license": SARAGA_LICENSE,
        "attribution": SARAGA_ATTRIBUTION,
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        "source_path": recording_path.relative_to(source / "dataset").as_posix(),
    }


def _configured_ids(path: Path) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {"carnatic": set(), "hindustani": set()}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"tradition", "recording_id"}.issubset(reader.fieldnames or []):
            raise ValueError("The sample ID file must have tradition and recording_id columns.")
        for row in reader:
            tradition = str(row.get("tradition") or "").strip().lower()
            recording_id = str(row.get("recording_id") or "").strip()
            if tradition not in selected:
                raise ValueError(f"Unsupported tradition in sample ID file: {tradition!r}.")
            if recording_id:
                selected[tradition].add(recording_id)
    return selected


def _repository_revision(source: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def import_saraga(
    source: str | Path = DEFAULT_SOURCE,
    database: str | Path = DEFAULT_DB,
    ids_file: str | Path = DEFAULT_IDS,
    *,
    import_all: bool = False,
) -> dict[str, Any]:
    """Import configured metadata from a local Saraga checkout, without downloading."""
    source, database, ids_file = Path(source).resolve(), Path(database).resolve(), Path(ids_file).resolve()
    for tradition in ("carnatic", "hindustani"):
        if not (source / "dataset" / tradition / "file_paths.csv").is_file():
            raise FileNotFoundError(f"Saraga mapping file missing for {tradition}; pass the root of a local MTG/saraga checkout with --source.")
    selected = {} if import_all else _configured_ids(ids_file)
    counts: dict[str, dict[str, int]] = {}
    imported_ids: list[str] = []
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.row_factory = sqlite3.Row
        ensure_schema(conn)
        for tradition in ("carnatic", "hindustani"):
            mapping_path = source / "dataset" / tradition / "file_paths.csv"
            with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not {"filepath", "mbid"}.issubset(reader.fieldnames or []):
                    raise ValueError(f"Unexpected Saraga file_paths.csv schema for {tradition}.")
                candidates = list(reader)
            total = imported = missing_metadata = 0
            imported_this_tradition: set[str] = set()
            for mapping in candidates:
                recording_id = str(mapping.get("mbid") or "").strip()
                if not import_all and recording_id not in selected[tradition]:
                    continue
                total += 1
                row = _metadata_row(source, tradition, mapping)
                if row is None:
                    missing_metadata += 1
                    continue
                columns = list(row)
                conn.execute(
                    f"INSERT INTO recordings ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                    f"ON CONFLICT(recording_id) DO UPDATE SET "
                    + ",".join(f"{column}=excluded.{column}" for column in columns if column != "recording_id"),
                    [row[column] for column in columns],
                )
                imported += 1
                imported_ids.append(row["recording_id"])
                imported_this_tradition.add(row["recording_id"])
            requested = selected.get(tradition, set()) if not import_all else set()
            counts[tradition] = {
                "configured_ids": len(requested),
                "selected_from_mapping": total,
                "imported_or_updated": imported,
                "missing_metadata_files": missing_metadata,
                "configured_ids_not_found": len(requested - imported_this_tradition),
            }
        conn.commit()
        total_recordings = int(conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0])
        audio_count = int(conn.execute("SELECT COUNT(*) FROM recordings WHERE audio_status='available_local'").fetchone()[0])
        language_count = int(conn.execute("SELECT COUNT(*) FROM recordings WHERE language IS NOT NULL AND TRIM(language)<>''").fetchone()[0])
        field_counts = {
            field: int(conn.execute(
                f"SELECT COUNT(*) FROM recordings WHERE {field} IS NOT NULL AND TRIM(CAST({field} AS TEXT))<>''"
            ).fetchone()[0])
            for field in ("raga", "tala", "laya", "form", "work", "tempo_bpm", "duration_ms")
        }
    return {
        "status": "ok",
        "source": SARAGA_SOURCE,
        "repository_revision": _repository_revision(source),
        "mode": "all_metadata" if import_all else "configured_sample",
        "imported_or_updated_this_run": len(imported_ids),
        "recordings_in_database": total_recordings,
        "audio_available_locally": audio_count,
        "language_values_available": language_count,
        "available_recording_fields": field_counts,
        "traditions": counts,
        "fields": [
            "recording_id", "title", "artist", "tradition", "release_info", "raga", "tala",
            "laya", "form", "work", "tempo_bpm", "tempo_annotations_json", "duration_ms",
            "language", "audio_path", "audio_status", "dataset_source", "dataset_license", "attribution",
        ],
        "notes": [
            "Spotify songs were preserved in the existing songs table.",
            "Only pre-existing local audio files are registered; this importer downloads no audio or metadata.",
            "Saraga language is not inferred from Hindustani or Carnatic tradition.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a small configured Saraga metadata sample into Tunely.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to a local MTG/saraga checkout; no network downloads are performed.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ids-file", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--all-metadata", action="store_true", help="Import all Saraga metadata records; audio is never downloaded.")
    parser.add_argument("--report", type=Path, help="Optional path for a JSON import report.")
    args = parser.parse_args()
    report = import_saraga(args.source, args.database, args.ids_file, import_all=args.all_metadata)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
