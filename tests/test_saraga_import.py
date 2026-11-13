"""Focused integration checks for metadata import using temporary fixtures."""

import csv
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from backend.recommender.saraga_import import import_saraga


class SaragaImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "saraga"
        self.database = self.root / "tunely.sqlite3"
        self.ids_file = self.root / "sample_ids.csv"
        self.carnatic_id = "6d4e58d1-e565-4987-b7a5-c63a4e9d3f90"
        self.hindustani_id = "b3a43a82-b3c3-49bf-bd3d-db561c1ec355"
        self._write_record("carnatic", "Release by Artist/Concert piece/Concert piece", self.carnatic_id, {
            "mbid": self.carnatic_id,
            "title": "Concert piece",
            "length": 2062000,
            "artists": [{"artist": {"mbid": "artist-id", "name": "Performer A"}, "lead": True}],
            "raaga": [{"mbid": "raga-id", "name": "Raga A"}],
            "taala": [],
            "form": [{"name": "Kriti"}],
            "work": [{"title": "Composition A"}],
            "concert": [{"title": "Release A"}],
            "album_artists": [{"name": "Performer A"}],
        })
        self._write_record("hindustani", "Release by Artist/Alap/Alap", self.hindustani_id, {
            "mbid": self.hindustani_id,
            "title": "Alap",
            "artists": [{"artist": {"name": "Performer B"}, "lead": True}],
            "raaga": [{"name": "Raga B"}],
            "taala": [{"name": "Tala B"}],
            "form": [],
            "work": [],
            "release": [{"title": "Release B"}],
        })
        self._write_mapping("carnatic", [["Release by Artist/Concert piece/Concert piece", self.carnatic_id]])
        self._write_mapping("hindustani", [["Release by Artist/Alap/Alap", self.hindustani_id]])
        self.ids_file.write_text(
            "tradition,recording_id\n"
            f"carnatic,{self.carnatic_id}\n"
            f"hindustani,{self.hindustani_id}\n",
            encoding="utf-8",
        )
        with closing(sqlite3.connect(self.database)) as conn, conn:
            conn.execute("CREATE TABLE songs(song_id TEXT PRIMARY KEY)")
            conn.execute("INSERT INTO songs VALUES ('existing-spotify-song')")

    def tearDown(self):
        self.temp.cleanup()

    def _write_mapping(self, tradition, rows):
        path = self.source / "dataset" / tradition / "file_paths.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["", "filepath", "mbid"])
            for index, (filepath, recording_id) in enumerate(rows):
                writer.writerow([index, filepath, recording_id])

    def _write_record(self, tradition, filepath, recording_id, metadata):
        recording_path = self.source / "dataset" / tradition / Path(filepath)
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        Path(str(recording_path) + ".json").write_text(json.dumps(metadata), encoding="utf-8")
        if tradition == "carnatic":
            Path(str(recording_path) + ".tempo-manual.txt").write_text(
                "340,170,2.471,14,2\n", encoding="utf-8"
            )

    def test_imports_selected_metadata_without_touching_songs_or_claiming_audio(self):
        report = import_saraga(self.source, self.database, self.ids_file)
        self.assertEqual(report["imported_or_updated_this_run"], 2)
        self.assertEqual(report["audio_available_locally"], 0)
        self.assertEqual(report["language_values_available"], 0)
        with closing(sqlite3.connect(self.database)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0], 1)
            row = conn.execute(
                "SELECT title, artist, tradition, raga, tempo_bpm, duration_ms, language, audio_status FROM recordings WHERE recording_id=?",
                (self.carnatic_id,),
            ).fetchone()
        self.assertEqual(row, ("Concert piece", "Performer A", "carnatic", "Raga A", 170.0, 2062000, None, "not_downloaded"))

    def test_import_is_idempotent_and_uses_hindustani_metadata_fields(self):
        import_saraga(self.source, self.database, self.ids_file)
        second = import_saraga(self.source, self.database, self.ids_file)
        self.assertEqual(second["recordings_in_database"], 2)
        with closing(sqlite3.connect(self.database)) as conn:
            row = conn.execute(
                "SELECT tradition, artist, raga, tala, tempo_bpm FROM recordings WHERE recording_id=?",
                (self.hindustani_id,),
            ).fetchone()
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM recordings").fetchone()[0], 2)
        self.assertEqual(row, ("hindustani", "Performer B", "Raga B", "Tala B", None))


if __name__ == "__main__":
    unittest.main()
