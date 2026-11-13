"""Tests for deduplicated Telugu dataset export without catalog mutation."""

import csv
import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.export_telugu_songs import export_telugu_songs
from backend.recommender.itunes_dataset import ensure_itunes_dataset_schema, upsert_itunes_tracks


class TeluguExportTests(unittest.TestCase):
    def test_local_and_multiple_search_results_are_deduplicated_and_metadata_is_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "tunely.sqlite3"
            output = Path(directory) / "telugu_songs.csv"
            conn = sqlite3.connect(database)
            ensure_itunes_dataset_schema(conn)
            upsert_itunes_tracks(conn, [
                {"kind": "song", "trackId": 100, "trackName": "తెలుగు పాట", "artistName": "Singer A",
                 "collectionName": "Album A", "releaseDate": "1998-02-01T00:00:00Z",
                 "primaryGenreName": "Classical", "artworkUrl100": "https://example.org/a.jpg",
                 "trackViewUrl": "https://music.apple.com/a", "trackTimeMillis": 180000},
                {"kind": "song", "trackId": 101, "trackName": "Latin transliteration", "artistName": "Singer B"},
            ], requested_language="Telugu", country="in")
            conn.commit()
            original_count = conn.execute("SELECT COUNT(*) FROM itunes_tracks").fetchone()[0]
            conn.close()

            class FakeClient:
                def search(self, term, _country, limit):
                    self.last_limit = limit
                    return [
                        {"kind": "song", "trackId": 100, "trackName": "తెలుగు పాట", "artistName": "Singer A"},
                        {"kind": "song", "trackId": 200, "trackName": "తెలుగు కొత్త పాట", "artistName": "Singer C",
                         "collectionName": "Album C", "releaseDate": "2025-01-01T00:00:00Z", "primaryGenreName": "Pop"},
                        {"kind": "song", "trackName": "No ID Song", "artistName": "Singer D", "collectionName": "Album D"},
                        {"kind": "song", "trackId": 200, "trackName": "తెలుగు కొత్త పాట", "artistName": "Singer C",
                         "collectionName": "Album C", "releaseDate": "2025-01-01T00:00:00Z", "primaryGenreName": "Pop"},
                    ]

            summary = export_telugu_songs(database=database, output=output, delay_seconds=0,
                                          queries=("Telugu songs", "తెలుగు పాటలు"), client=FakeClient())
            with output.open(encoding="utf-8-sig", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 4)
            self.assertEqual(summary["songs_from_existing_database"], 2)
            self.assertEqual(summary["additional_songs_from_itunes"], 2)
            self.assertEqual(summary["duplicate_records_removed"], 6)
            self.assertEqual(summary["songs_with_missing_metadata"], 4)
            self.assertEqual(summary["missing_metadata_by_field"]["Language"], 2)
            self.assertEqual(summary["api"]["limit_per_query"], 200)
            verified = next(row for row in rows if row["iTunes Track ID"] == "100")
            self.assertEqual(verified["Language"], "Telugu")
            self.assertIn("stored database estimate", verified["Language Status"])
            self.assertEqual(verified["Release Year"], "1998")
            self.assertEqual(verified["Duration (ms)"], "180000")
            unknown = next(row for row in rows if row["iTunes Track ID"] == "101")
            self.assertEqual(unknown["Language"], "Unknown")
            self.assertEqual(unknown["Release Year"], "")
            check = sqlite3.connect(database)
            self.assertEqual(check.execute("SELECT COUNT(*) FROM itunes_tracks").fetchone()[0], original_count)
            check.close()


if __name__ == "__main__":
    unittest.main()
