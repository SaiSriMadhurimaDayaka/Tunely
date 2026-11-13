"""Integration tests for the manually reviewed catalog import and ranking."""

import json
import sqlite3
import tempfile
import unittest
import zipfile
from xml.sax.saxutils import escape
from pathlib import Path

from backend.app import create_app
from backend.recommender.ranking import DEFAULT_DB
from backend.recommender.reviewed_catalog import (
    APPROVED_GENRES, ensure_reviewed_catalog_schema, genre_targets,
    import_reviewed_workbook, reviewed_catalog_statistics,
)


HEADERS = ("Song Title", "Artist Name", "Album Name", "Language", "Release Date", "Release Year",
           "iTunes Track ID", "Primary Genre", "Artwork URL", "Track URL", "Preview URL", "Duration (ms)",
           "Language Status", "Discovered By", "Genre Classification Evidence")


def write_test_workbook(path, records):
    def column_name(index):
        result = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            result = chr(65 + remainder) + result
        return result

    rows = [HEADERS, *[tuple(record.get(header, "") for header in HEADERS) for record in records]]
    xml_rows = []
    for row_number, values in enumerate(rows, 1):
        cells = []
        for column, value in enumerate(values, 1):
            ref = f"{column_name(column)}{row_number}"
            cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    workbook_xml = ('<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                    '<sheets><sheet name="Songs" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels_xml = ('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                'Target="worksheets/sheet1.xml"/></Relationships>')
    sheet_xml = ('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                 f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


class ReviewedCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "reviewed.sqlite3"
        conn = sqlite3.connect(self.database)
        try:
            conn.execute("""CREATE TABLE songs (
                song_id TEXT PRIMARY KEY, track_id TEXT, song_name TEXT NOT NULL, artist TEXT NOT NULL,
                album_name TEXT, release_date TEXT, release_year INTEGER, popularity INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER NOT NULL DEFAULT 0, duration_minutes REAL NOT NULL DEFAULT 0,
                album_type TEXT, cover_image TEXT, explicit INTEGER, track_genre TEXT,
                available_features TEXT NOT NULL DEFAULT '', danceability REAL, energy REAL, acousticness REAL,
                instrumentalness REAL, valence REAL, speechiness REAL, liveness REAL, tempo REAL, loudness REAL
            )""")
            conn.commit()
        finally:
            conn.close()
        self.app = create_app(self.database, testing=True)
        source = sqlite3.connect(DEFAULT_DB)
        target = sqlite3.connect(self.database)
        try:
            columns = [row[1] for row in source.execute("PRAGMA table_info(reviewed_songs)")]
            names = ",".join(f'"{column}"' for column in columns)
            values = source.execute(f"SELECT {names} FROM reviewed_songs").fetchall()
            target.executemany(f"INSERT INTO reviewed_songs({names}) VALUES({','.join('?' for _ in columns)})", values)
            target.commit()
            self.stats = reviewed_catalog_statistics(target)
        finally:
            source.close()
            target.close()
        self.client = self.app.test_client()
        self.headers = {"X-Session-ID": "reviewed-test"}

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_imports_all_rows_idempotently_and_preserves_manual_values(self):
        self.assertEqual(self.stats["song_count"], 1070)
        self.assertEqual(self.stats["language_counts"]["Telugu"], 457)
        self.assertEqual(self.stats["language_counts"]["Hindi"], 449)
        self.assertEqual(self.stats["language_counts"]["Malayalam"], 0)
        self.assertEqual(self.stats["language_counts"]["Kannada"], 0)
        self.assertTrue(all(self.stats["genre_counts"][genre] > 0 for genre in APPROVED_GENRES))
        conn = sqlite3.connect(self.database)
        try:
            self.assertEqual(conn.execute("SELECT COUNT(DISTINCT itunes_track_id) FROM reviewed_songs").fetchone()[0], 1070)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM reviewed_songs WHERE title IS NULL OR artist_name IS NULL OR language IS NULL OR artwork_url IS NULL OR genres_json IS NULL").fetchone()[0], 0)
            row = conn.execute("SELECT primary_genre_raw,genres_json FROM reviewed_songs WHERE primary_genre_raw='Devotional; Old' LIMIT 1").fetchone()
            self.assertEqual(row[0], "Devotional; Old")
            self.assertIn("Devotional", json.loads(row[1]))
            self.assertIn("Old", json.loads(row[1]))
        finally:
            conn.close()

    def test_excel_import_is_repeatable_and_normalizes_genres_without_losing_source_text(self):
        workbook = Path(self.temp_dir.name) / "reviewed-fixture.xlsx"
        base = {"Song Title": "Reviewed Track", "Artist Name": "Artist", "Album Name": "Album",
                "Language": "telugu ", "Release Date": "2009-01-01", "Release Year": 2009,
                "iTunes Track ID": 98765, "Primary Genre": "devotional; Old",
                "Artwork URL": "https://example.test/art.jpg", "Track URL": "https://example.test/track",
                "Preview URL": "https://example.test/preview", "Duration (ms)": 180000,
                "Language Status": "reviewed", "Discovered By": "manual review",
                "Genre Classification Evidence": "reviewer label"}
        newer = dict(base, **{"Song Title": "New Track", "Artist Name": "Artist 2", "Release Year": 2021,
                              "iTunes Track ID": 98766, "Primary Genre": "party; old"})
        duplicate = dict(base, **{"Song Title": "Ignored duplicate"})
        write_test_workbook(workbook, [base, newer, duplicate])
        db = Path(self.temp_dir.name) / "import.sqlite3"
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            ensure_reviewed_catalog_schema(conn)
            first = import_reviewed_workbook(conn, workbook)
            second = import_reviewed_workbook(conn, workbook)
            rows = conn.execute("SELECT language,primary_genre_raw,genres_json FROM reviewed_songs ORDER BY itunes_track_id").fetchall()
        finally:
            conn.close()
        self.assertEqual(first["inserted"], 2)
        self.assertEqual(first["duplicates_removed"], 1)
        self.assertEqual(second["inserted"], 0)
        self.assertEqual(second["updated"], 2)
        self.assertEqual(rows[0]["language"], "Telugu")
        self.assertEqual(rows[0]["primary_genre_raw"], "devotional; Old")
        self.assertEqual(json.loads(rows[0]["genres_json"]), ["Old", "Devotional"])
        self.assertEqual(json.loads(rows[1]["genres_json"]), ["Party"])

    def test_old_category_obeys_fixed_release_cutoff_without_losing_other_labels(self):
        conn = sqlite3.connect(self.database)
        try:
            newer_old = conn.execute("SELECT COUNT(*) FROM reviewed_songs WHERE release_year>=2010 AND genres_json LIKE '%Old%'").fetchone()[0]
            older_missing = conn.execute("SELECT COUNT(*) FROM reviewed_songs WHERE release_year<=2009 AND genres_json NOT LIKE '%Old%'").fetchone()[0]
            multi = conn.execute("SELECT genres_json FROM reviewed_songs WHERE release_year<=2009 AND primary_genre_raw='Devotional; Old' LIMIT 1").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(newer_old, 0)
        self.assertEqual(older_missing, 0)
        self.assertEqual(json.loads(multi), ["Old", "Devotional"])

    def test_language_mood_intensity_and_refresh_use_reviewed_rows(self):
        languages = self.client.get("/api/discovery/languages").json["data"]
        self.assertEqual([row["language"] for row in languages],
                         ["Telugu", "Hindi", "Tamil", "Malayalam", "Kannada", "English"])
        self.assertEqual(languages[3]["catalog_song_count"], 0)
        for mood, expected in (("sad", "Sad"), ("happy", "Party"), ("romantic", "Romantic"), ("nostalgic", "Old")):
            response = self.client.post("/api/mood-recommendations", headers=self.headers, json={
                "language": "Telugu", "mood": mood, "intensity": "Moderate", "limit": 8})
            self.assertEqual(response.status_code, 200)
            data = response.json["data"]
            self.assertEqual(data["api_source"], "reviewed_dataset")
            self.assertTrue(data["recommendations"])
            self.assertTrue(all(item["language"] == "Telugu" for item in data["recommendations"]))
            self.assertTrue(all(expected in item["music_categories"] for item in data["recommendations"]))
        for mood, expected_any in (("energetic", {"Party", "Mass"}),
                                   ("motivated", {"Mass", "Party"}),
                                   ("calm", {"Devotional", "Romantic", "Sad"}),
                                   ("stressed", {"Devotional", "Sad", "Romantic"})):
            response = self.client.post("/api/mood-recommendations", headers={"X-Session-ID": f"mood-{mood}"}, json={
                "language": "Hindi", "mood": mood, "intensity": "Intense", "limit": 8})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(any(expected_any.intersection(item["music_categories"])
                                for item in response.json["data"]["recommendations"]))
        for language in ("Hindi", "Tamil", "English"):
            result = self.client.post("/api/mood-recommendations", headers={"X-Session-ID": f"lang-{language}"}, json={
                "language": language, "mood": "sad", "intensity": "Moderate", "limit": 3})
            self.assertTrue(all(item["language"] == language for item in result.json["data"]["recommendations"]))
        self.assertGreater(genre_targets("energetic", "", "intense")["Party"],
                           genre_targets("energetic", "", "mild")["Party"])
        first = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Hindi", "mood": "happy", "intensity": "Mild", "limit": 8}).json["data"]["recommendations"]
        next_page = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Hindi", "mood": "happy", "intensity": "Mild", "refresh": True,
            "exclude_track_ids": [song["itunes_track_id"] for song in first], "limit": 8}).json["data"]["recommendations"]
        self.assertTrue(set(item["itunes_track_id"] for item in first).isdisjoint(
            item["itunes_track_id"] for item in next_page))

    def test_intent_ml_scores_personalize_recommendations_and_keep_catalog_language(self):
        model = self.app.extensions.get("tunely_intent_model")
        self.assertIsNotNone(model)
        devotional = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Telugu", "mood": "neutral", "intensity": "Moderate",
            "preference": "devotional prayer songs", "limit": 5,
        })
        self.assertEqual(devotional.status_code, 200)
        devotional_data = devotional.json["data"]
        self.assertEqual(devotional_data["recommendation_model"], "tfidf_ovr_logistic_regression")
        self.assertGreater(devotional_data["genre_relevance_scores"]["Devotional"],
                           devotional_data["genre_relevance_scores"]["Sad"])
        self.assertTrue(devotional_data["recommendations"])
        self.assertTrue(all(song["language"] == "Telugu" for song in devotional_data["recommendations"]))

        heartbreak = model.predict_scores("heartbreak and separation songs", "neutral", "moderate")
        self.assertGreater(heartbreak["Sad"], heartbreak["Devotional"])
        for mood in ("happy", "sad", "calm", "energetic", "stressed", "romantic", "neutral", "motivated", "nostalgic"):
            for intensity in ("Mild", "Moderate", "Intense"):
                with self.subTest(mood=mood, intensity=intensity):
                    response = self.client.post("/api/mood-recommendations", headers={
                        "X-Session-ID": f"ml-{mood}-{intensity.lower()}"}, json={
                        "language": "Telugu", "mood": mood, "intensity": intensity,
                        "preference": "I want heartfelt music", "limit": 2,
                    })
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json["data"]["recommendation_model"],
                                     "tfidf_ovr_logistic_regression")

    def test_empty_language_and_saved_track_workflow(self):
        empty = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Kannada", "mood": "sad", "intensity": "Intense", "limit": 8})
        self.assertEqual(empty.status_code, 200)
        self.assertEqual(empty.json["data"]["recommendations"], [])
        self.assertIn("no songs labeled Kannada", empty.json["data"]["message"])
        response = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Telugu", "mood": "sad", "intensity": "Mild", "limit": 1})
        song = response.json["data"]["recommendations"][0]
        saved = self.client.post("/api/itunes-favorites", headers=self.headers,
                                 json={"track_id": song["itunes_track_id"]})
        self.assertEqual(saved.status_code, 201)
        favorite = self.client.get("/api/itunes-favorites", headers=self.headers).json["data"][0]
        self.assertEqual(favorite["song_id"], song["song_id"])

    def test_direct_reviewed_genre_filters_for_the_seven_requested_combinations(self):
        cases = (
            ("Telugu", "Sad", "Mild", True),
            ("Telugu", "Romantic", "Moderate", True),
            ("Hindi", "Party", "Intense", True),
            ("Tamil", "Devotional", "Mild", True),
            ("Malayalam", "Old", "Moderate", False),
            ("Kannada", "Mass", "Intense", False),
            ("English", "Romantic", "Moderate", True),
        )
        for language, genre, intensity, has_expected_rows in cases:
            with self.subTest(language=language, genre=genre):
                response = self.client.post("/api/mood-recommendations", headers={
                    "X-Session-ID": f"category-{language}-{genre}"}, json={
                    "language": language, "mood": "neutral", "genre": genre,
                    "intensity": intensity, "limit": 8})
                self.assertEqual(response.status_code, 200)
                data = response.json["data"]
                self.assertEqual(data["genre"], genre)
                songs = data["recommendations"]
                self.assertEqual(bool(songs), has_expected_rows)
                self.assertTrue(all(language == song["language"] for song in songs))
                self.assertTrue(all(genre in song["music_categories"] for song in songs))
                if not has_expected_rows:
                    self.assertTrue(data["message"])

        explicit = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Telugu", "mood": "neutral", "genre": " old ",
            "intensity": "Moderate", "limit": 8})
        self.assertEqual(explicit.status_code, 200)
        self.assertTrue(all("Old" in song["music_categories"]
                            for song in explicit.json["data"]["recommendations"]))


if __name__ == "__main__":
    unittest.main()
