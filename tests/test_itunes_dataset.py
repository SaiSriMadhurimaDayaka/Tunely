"""Tests for incremental, normalized iTunes dataset persistence."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from backend.collect_itunes_dataset import LANGUAGES, collect_dataset
from backend.recommender.itunes_dataset import (
    dataset_statistics, ensure_itunes_dataset_schema, local_itunes_items, upsert_itunes_tracks,
)


class ITunesDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "itunes.sqlite3"
        self.conn = sqlite3.connect(self.database)
        self.conn.row_factory = sqlite3.Row
        ensure_itunes_dataset_schema(self.conn)

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_incremental_storage_dedup_normalization_and_metadata(self):
        real_item = {"kind": "song", "trackId": 771, "trackName": "  తెలుగు   పాట ",
                     "artistName": "Singer   A", "collectionName": "Album A",
                     "primaryGenreName": "Dance", "releaseDate": "1998-02-01T00:00:00Z",
                     "trackTimeMillis": 180000, "artworkUrl100": "http://art.example/a.jpg",
                     "trackViewUrl": "https://music.apple.com/in/song/771",
                     "previewUrl": "https://audio.example/771.m4a", "collectionId": 99}
        counts = upsert_itunes_tracks(self.conn, [real_item, dict(real_item), {"kind": "song"}],
                                      requested_language="Telugu", country="in")
        self.assertEqual(counts, {"inserted": 1, "updated": 0, "invalid": 1})
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM itunes_tracks WHERE track_id='771'").fetchone()
        self.assertEqual(row["artist_normalized"], "singer a")
        self.assertEqual(row["genre_normalized"], "dance")
        self.assertEqual(row["release_year"], 1998)
        self.assertEqual(row["language"], "Telugu")
        self.assertEqual(row["country"], "in")
        self.assertEqual(row["collection_id"], "99")
        self.assertEqual(row["artwork_url"], "https://art.example/a.jpg")
        self.assertIsNotNone(row["preview_url"])
        self.assertEqual(__import__("json").loads(row["category_ids_json"]), ["party", "old"])

        update = upsert_itunes_tracks(self.conn, [{"kind": "song", "trackId": 771,
            "trackName": "తెలుగు పాట", "artistName": "Singer A"}], requested_language="Telugu", country="in")
        self.assertEqual(update["updated"], 1)
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM itunes_tracks").fetchone()[0], 1)
        self.assertEqual(len(local_itunes_items(self.conn, "Telugu")), 1)
        stats = dataset_statistics(self.conn)
        self.assertEqual(stats["track_count"], 1)
        self.assertEqual(stats["requested_language_counts"]["Telugu"], 1)
        self.assertEqual(stats["category_counts"]["party"], 1)

    def test_uncertain_language_and_optional_metadata_remain_unknown(self):
        item = {"kind": "song", "trackId": 772, "trackName": "Latin transliteration",
                "artistName": "Singer B"}
        upsert_itunes_tracks(self.conn, [item], requested_language="Hindi", country="in")
        row = self.conn.execute("SELECT language,language_confidence,release_year,preview_url FROM itunes_tracks WHERE track_id='772'").fetchone()
        self.assertIsNone(row["language"])
        self.assertEqual(row["language_confidence"], 0)
        self.assertIsNone(row["release_year"])
        self.assertIsNone(row["preview_url"])

    def test_collector_uses_distinct_queries_and_skips_recently_collected_terms(self):
        class FakeClient:
            calls = []
            def search(self, term, country, limit):
                self.calls.append((term, country, limit))
                return [{"kind": "song", "trackId": 9001, "trackName": "తెలుగు పాట", "artistName": "Singer"}]

        fake = FakeClient()
        first = collect_dataset(database=self.database, languages=("Telugu",), country="in",
                                limit=200, delay_seconds=0, client=fake)
        self.assertEqual(first["requests"], 7)
        self.assertEqual(first["dataset"]["track_count"], 1)
        self.assertTrue(all(call[1:] == ("in", 200) for call in fake.calls))
        self.assertEqual(len({call[0] for call in fake.calls}), 7)
        second = collect_dataset(database=self.database, languages=("Telugu",), country="in",
                                 limit=200, delay_seconds=0, client=fake)
        self.assertEqual(second["requests"], 0)
        self.assertEqual(second["skipped_queries"], 7)
        self.assertEqual(len(fake.calls), 7)

    def test_all_eight_target_languages_are_collected_without_assuming_verified_language(self):
        class EightLanguageClient:
            calls = []
            def search(self, term, country, limit):
                self.calls.append(term)
                language = next(name for name in LANGUAGES if term.startswith(name))
                track_id = LANGUAGES.index(language) + 1000
                script_title = {"Telugu": "తెలుగు పాట", "Hindi": "हिंदी गीत", "Tamil": "தமிழ் பாடல்",
                                "Malayalam": "മലയാളം പാട്ട്", "Kannada": "ಕನ್ನಡ ಹಾಡು",
                                "Punjabi": "ਪੰਜਾਬੀ ਗੀਤ", "Bengali": "বাংলা গান"}.get(language, "Song Without Language Label")
                return [{"kind": "song", "trackId": track_id, "trackName": script_title}]

        fake = EightLanguageClient()
        result = collect_dataset(database=self.database, languages=LANGUAGES, delay_seconds=0, client=fake)
        self.assertEqual(result["requests"], 56)
        self.assertEqual(result["dataset"]["track_count"], 8)
        self.assertEqual(set(result["dataset"]["requested_language_counts"]), set(LANGUAGES))
        self.assertIsNone(self.conn.execute("SELECT language FROM itunes_tracks WHERE track_id='1005'").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
