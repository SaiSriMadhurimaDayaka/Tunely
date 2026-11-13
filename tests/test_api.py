"""Integration checks for the Flask API using an isolated SQLite catalog."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from backend.app import create_app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "api.sqlite3"
        self._seed_catalog()
        self.app = create_app(self.database, testing=True)
        self._seed_saraga()
        self.client = self.app.test_client()
        self.headers = {"X-Session-ID": "test-user"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _seed_saraga(self) -> None:
        conn = sqlite3.connect(self.database)
        try:
            conn.executemany("""INSERT INTO recordings
                (recording_id, title, artist, tradition, release_info, raga, tala, laya, form, work,
                 tempo_bpm, tempo_annotations_json, duration_ms, language, audio_path, audio_status,
                 dataset_source, dataset_license, attribution, metadata_json, source_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
                ("saraga-a", "Raga A piece", "Performer A", "carnatic", "Concert A", "Raga A; Rāga A", "Tala A", None, "Kriti", "Work A", 120.0, "[]", 180000, None, None, "not_downloaded", "MTG Saraga", "CC BY-NC 4.0", "Saraga/MTG", "{}", "carnatic/concert-a/raga-a"),
                ("saraga-b", "Raga B piece", "Performer B", "hindustani", "Concert B", "Raga B", "Tala B", "vilambit", "Khayal", "Work B", None, "[]", 210000, None, None, "not_downloaded", "MTG Saraga", "CC BY-NC 4.0", "Saraga/MTG", "{}", "hindustani/concert-b/raga-b"),
            ])
            conn.commit()
        finally:
            conn.close()

    def _seed_catalog(self) -> None:
        conn = sqlite3.connect(self.database)
        try:
            conn.execute("""CREATE TABLE songs (
                song_id TEXT PRIMARY KEY, track_id TEXT UNIQUE, song_name TEXT NOT NULL,
                artist TEXT NOT NULL, album_name TEXT, release_date TEXT, release_year INTEGER,
                popularity INTEGER NOT NULL, duration_ms INTEGER NOT NULL, duration_minutes REAL NOT NULL,
                album_type TEXT, cover_image TEXT, explicit INTEGER, track_genre TEXT,
                available_features TEXT NOT NULL,
                danceability REAL, energy REAL, acousticness REAL, instrumentalness REAL,
                valence REAL, speechiness REAL, liveness REAL, tempo REAL, loudness REAL
            )""")
            conn.executemany("""INSERT INTO songs VALUES
                (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, NULL, NULL, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
                ("song-a", "track-a", "Quiet Light", "Artist A", "Album A", 65, 180000, 3.0, "acoustic", "energy,acousticness,valence", 0.25, 0.75, 0.8, 0.4, 0.6, 0.05, 0.1, 85.0, -12.0),
                ("song-b", "track-b", "Bright Move", "Artist B", "Album B", 88, 200000, 3.333, "dance", "energy,danceability,valence", 0.82, 0.2, 0.1, 0.0, 0.8, 0.04, 0.12, 125.0, -5.0),
            ])
            conn.commit()
        finally:
            conn.close()

    def test_health_songs_and_recommendations(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json["data"]["song_count"], 2)
        songs = self.client.get("/api/songs?limit=1")
        self.assertEqual(songs.status_code, 200)
        self.assertEqual(songs.json["meta"]["total"], 2)
        response = self.client.post("/api/recommendations", headers=self.headers, json={
            "mood": "calm", "goal": "relax and unwind", "limit": 2,
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["data"]["recommendations"]), 2)
        self.assertTrue(response.json["data"]["recommendations"][0]["recommendation_id"])
        self.assertIn(response.json["data"]["strategy"], {"features", "exploration"})
        self.assertTrue(response.json["data"]["recommendations"][0]["youtube_music_search_url"].startswith("https://music.youtube.com/search?"))

    def test_mood_intensity_accepts_three_levels_and_rejects_others(self) -> None:
        for level in (1, 2, 3):
            response = self.client.post("/api/recommendations", headers={"X-Session-ID": f"level-{level}"}, json={
                "mood": "happy", "mood_level": level, "limit": 1,
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["data"]["context"]["mood_level"], level)
        invalid = self.client.post("/api/recommendations", headers=self.headers, json={
            "mood": "happy", "mood_level": 4, "limit": 1,
        })
        self.assertEqual(invalid.status_code, 400)

    def test_itunes_discovery_cache_saved_tracks_feedback_and_history(self) -> None:
        class FakeITunesClient:
            calls = []
            def search(self, term, country, limit):
                self.calls.append((term, country, limit))
                return [{"kind": "song", "trackId": 12345, "trackName": "తెలుగు calm melody",
                         "artistName": "Artist Name", "collectionName": "Album One",
                         "primaryGenreName": "Ambient", "artworkUrl100": "https://is1-ssl.mzstatic.com/artwork.jpg",
                         "trackTimeMillis": 190000, "trackViewUrl": "https://music.apple.com/in/album/song/12345"}]

        fake = FakeITunesClient()
        self.app.extensions["tunely_itunes_client"] = fake
        payload = {"language": "Telugu", "mood": "calm", "intensity": "Mild", "limit": 1}
        response = self.client.post("/api/mood-recommendations", headers=self.headers, json=payload)
        self.assertEqual(response.status_code, 200)
        result = response.json["data"]
        self.assertEqual(len(result["recommendations"]), 1)
        song = result["recommendations"][0]
        self.assertEqual(song["itunes_track_id"], "12345")
        self.assertEqual(song["artist"], "Artist Name")
        self.assertEqual(song["language"], "Telugu")
        self.assertFalse(song["language_verified"])
        self.assertEqual(song["ai_genre"], "gentle ambient")
        self.assertIsNone(song.get("youtube_video_id"))
        self.assertEqual(urlsplit(song["youtube_music_search_url"]).netloc, "music.youtube.com")
        self.assertEqual(parse_qs(urlsplit(song["youtube_music_search_url"]).query)["q"], ["తెలుగు calm melody Artist Name"])
        self.assertEqual(fake.calls[0][1:], ("in", 50))

        second = self.client.post("/api/mood-recommendations", headers=self.headers, json=payload)
        self.assertEqual(len(second.json["data"]["recommendations"]), 1)
        self.assertGreater(len(fake.calls), 1)  # Context rotation adds a new query beyond the cached base.

        saved = self.client.post("/api/itunes-favorites", headers=self.headers, json={"track_id": "12345"})
        self.assertEqual(saved.status_code, 201)
        favorites = self.client.get("/api/itunes-favorites", headers=self.headers).json["data"]
        self.assertEqual(favorites[0]["itunes_track_id"], "12345")
        feedback = self.client.post("/api/itunes-feedback", headers=self.headers, json={
            "recommendation_id": song["recommendation_id"], "rating": "Loved it",
        })
        self.assertEqual(feedback.status_code, 201)
        history = self.client.get("/api/history?limit=10", headers=self.headers).json["data"]
        itunes_history = next(item for item in history if item.get("itunes_track_id") == "12345")
        self.assertEqual(itunes_history["rating"], "loved it")
        removed = self.client.delete("/api/itunes-favorites/12345", headers=self.headers)
        self.assertEqual(removed.status_code, 200)

    def test_itunes_discovery_needs_no_api_key_and_validates_inputs(self) -> None:
        class EmptyITunesClient:
            def search(self, term, country, limit):
                return []

        self.app.extensions["tunely_itunes_client"] = EmptyITunesClient()
        response = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Telugu", "mood": "sad", "intensity": "Intense",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["api_source"], "itunes")
        self.assertEqual(data["recommendations"], [])
        self.assertIn("no songs", data["message"])
        bad = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Klingon", "mood": "sad", "intensity": "Intense",
        })
        self.assertEqual(bad.status_code, 400)

    def test_itunes_results_without_optional_metadata_are_kept_for_frontend_cards(self) -> None:
        class MinimalITunesClient:
            def search(self, term, country, limit):
                return [{"kind": "song", "trackId": "98765", "trackName": "Instrumental Study No. 1"}]

        self.app.extensions["tunely_itunes_client"] = MinimalITunesClient()
        response = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Telugu", "mood": "calm", "intensity": "Mild", "limit": 1,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["api_source"], "itunes")
        self.assertEqual(len(data["recommendations"]), 1)
        card = data["recommendations"][0]
        self.assertEqual(card["song_name"], "Instrumental Study No. 1")
        self.assertEqual(card["language"], "Unknown")
        self.assertIsNone(card["cover_image"])
        self.assertTrue(card["ai_genre"])
        self.assertEqual(card["requested_language"], "Telugu")
        self.assertIn("youtube_music_search_url", card)

    def test_category_discovery_uses_mood_and_goal_without_dropping_unlabeled_tracks(self) -> None:
        class CategorizedClient:
            calls = []
            def search(self, term, country, limit):
                self.calls.append(term)
                return [
                    {"kind": "song", "trackId": 501, "trackName": "Dance Track", "artistName": "Artist",
                     "primaryGenreName": "Dance", "releaseDate": "2025-01-01T00:00:00Z"},
                    {"kind": "song", "trackId": 502, "trackName": "Quiet Track", "artistName": "Artist",
                     "primaryGenreName": "Classical", "releaseDate": "1990-01-01T00:00:00Z"},
                ]

        fake = CategorizedClient()
        self.app.extensions["tunely_itunes_client"] = fake
        categories = self.client.get("/api/discovery/categories")
        self.assertEqual(categories.status_code, 200)
        self.assertEqual(len(categories.json["data"]), 7)
        response = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Telugu", "mood": "happy", "intensity": "Moderate", "goal": "energize",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(len(data["recommendations"]), 2)
        self.assertEqual(data["recommendations"][0]["itunes_track_id"], "501")
        self.assertIn("Party Songs", data["recommendations"][0]["music_categories"])
        self.assertEqual(fake.calls[0], "Telugu songs")
        self.assertGreaterEqual(len(fake.calls), 2)
        self.assertEqual(data["inferred_category_labels"][:2], ["Party Songs", "Mass Songs"])

    def test_refresh_rotates_search_and_avoids_recent_tracks_with_artist_diversity(self) -> None:
        class ExpandingClient:
            calls = []
            def search(self, term, country, limit):
                self.calls.append(term)
                if term == "Telugu songs":
                    start = 1
                elif "artist" in term.casefold() or "album" in term.casefold():
                    start = 21
                else:
                    start = 11
                return [{"kind": "song", "trackId": index, "trackName": f"Track {index}",
                         "artistName": f"Artist {index}", "collectionName": f"Album {index}",
                         "primaryGenreName": "Dance", "releaseDate": f"202{index % 6}-01-01T00:00:00Z"}
                        for index in range(start, start + 10)]

        fake = ExpandingClient()
        self.app.extensions["tunely_itunes_client"] = fake
        payload = {"language": "Telugu", "mood": "happy", "intensity": "Moderate", "goal": "feel-good", "limit": 8}
        first = self.client.post("/api/mood-recommendations", headers=self.headers, json=payload).json["data"]
        first_ids = {item["itunes_track_id"] for item in first["recommendations"]}
        self.assertEqual(len(first_ids), 8)
        self.assertEqual(len({item["artist"] for item in first["recommendations"]}), 8)
        before_refresh = len(fake.calls)
        refresh = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            **payload, "refresh": True,
            "exclude_track_ids": [item["itunes_track_id"] for item in first["recommendations"]],
        }).json["data"]
        refresh_ids = {item["itunes_track_id"] for item in refresh["recommendations"]}
        self.assertEqual(len(refresh_ids), 8)
        self.assertFalse(first_ids & refresh_ids)
        self.assertGreater(len(fake.calls), before_refresh)
        self.assertNotEqual(first["search_queries"], refresh["search_queries"])

    def test_discovered_itunes_tracks_are_persisted_and_reused_from_local_dataset(self) -> None:
        class CatalogClient:
            calls = []
            def search(self, term, country, limit):
                self.calls.append((term, country, limit))
                return [{"kind": "song", "trackId": index, "trackName": f"తెలుగు పాట {index}",
                         "artistName": f"Singer {index}", "collectionName": f"Album {index}",
                         "primaryGenreName": "Dance", "releaseDate": "2025-01-01T00:00:00Z",
                         "trackTimeMillis": 180000}
                        for index in range(700, 760)]

        fake = CatalogClient()
        self.app.extensions["tunely_itunes_client"] = fake
        payload = {"language": "Telugu", "mood": "happy", "intensity": "Moderate", "goal": "feel-good", "limit": 8}
        first = self.client.post("/api/mood-recommendations", headers=self.headers, json=payload).json["data"]
        self.assertEqual(len(first["recommendations"]), 8)
        self.assertEqual(first["discovery_source"], "local_dataset_and_itunes_api")
        self.assertEqual(self.client.get("/api/itunes-dataset/stats").json["data"]["track_count"], 60)
        call_count = len(fake.calls)
        second = self.client.post("/api/mood-recommendations", headers=self.headers, json=payload).json["data"]
        self.assertEqual(second["discovery_source"], "local_dataset")
        self.assertEqual(second["local_dataset_count"], 60)
        self.assertEqual(len(fake.calls), call_count)
        self.assertFalse({row["itunes_track_id"] for row in first["recommendations"]}
                         & {row["itunes_track_id"] for row in second["recommendations"]})

    def test_discovery_language_list_and_language_specific_search_targets(self) -> None:
        supported = self.client.get("/api/discovery/languages")
        self.assertEqual(supported.status_code, 200)
        options = {item["language"]: item for item in supported.json["data"]}
        for language in ("Telugu", "Hindi", "Tamil", "Kannada", "Malayalam", "Bengali", "Marathi", "Punjabi", "English"):
            with self.subTest(language=language):
                self.assertTrue(options[language]["search_supported"])

        class LanguageAwareClient:
            calls = []
            def search(self, query, country, limit):
                self.calls.append((query, country, limit))
                if "Telugu" in query:
                    title = "తెలుగు పాట"
                    track_id = 101
                else:
                    title = "हिंदी गीत"
                    track_id = 102
                return [{"kind": "song", "trackId": track_id, "trackName": title,
                         "artistName": "Artist", "primaryGenreName": "Ambient"}]

        fake = LanguageAwareClient()
        self.app.extensions["tunely_itunes_client"] = fake
        for language, expected_script in (("Telugu", "తెలుగు"), ("Hindi", "हिंदी")):
            response = self.client.post("/api/mood-recommendations", headers={"X-Session-ID": language}, json={
                "language": language, "mood": "calm", "intensity": "Moderate", "limit": 1,
            })
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["data"]["language"], language)
            self.assertIn(expected_script, response.json["data"]["recommendations"][0]["song_name"])
            self.assertTrue(any(language in query and country == "in" for query, country, _ in fake.calls))

    def test_discovery_rejects_missing_required_selections(self) -> None:
        for payload in ({"mood": "calm", "intensity": "Mild"},
                        {"language": "Telugu", "intensity": "Mild"},
                        {"language": "Telugu", "mood": "calm"}):
            with self.subTest(payload=payload):
                response = self.client.post("/api/mood-recommendations", headers=self.headers, json=payload)
                self.assertEqual(response.status_code, 400)

    def test_itunes_api_rate_limit_returns_helpful_error(self) -> None:
        from backend.recommender.itunes_discovery import ITunesDiscoveryError

        class RateLimitedClient:
            def search(self, *args, **kwargs):
                raise ITunesDiscoveryError("The music catalog is receiving too many requests. Please try again shortly.", "rate_limited")

        self.app.extensions["tunely_itunes_client"] = RateLimitedClient()
        response = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "Hindi", "mood": "happy", "intensity": "Moderate",
        })
        self.assertEqual(response.status_code, 200)
        data = response.json["data"]
        self.assertEqual(data["error_code"], "rate_limited")
        self.assertEqual(data["recommendations"], [])
        self.assertIn("try again shortly", data["message"])

    def test_song_specific_youtube_link_metadata_and_missing_mood_fields(self) -> None:
        conn = sqlite3.connect(self.database)
        try:
            conn.execute("UPDATE songs SET song_name=?, artist=?, language=?, mood=?, youtube_music_url=? WHERE song_id=?",
                         ("Don't Stop & Go", "Björk & Co.", "Icelandic", "calm", "https://music.youtube.com/watch?v=verified123", "song-a"))
            conn.commit()
        finally:
            conn.close()
        response = self.client.post("/api/recommendations", headers=self.headers, json={
            "mood": "calm", "goal": "relax and unwind", "limit": 2,
        })
        tracks = response.json["data"]["recommendations"]
        linked = next(song for song in tracks if song["song_id"] == "song-a")
        self.assertEqual(linked["youtube_music_url"], "https://music.youtube.com/watch?v=verified123")
        self.assertEqual(linked["language"], "Icelandic")
        self.assertEqual(linked["mood"], "calm")
        self.assertIn("%26", linked["youtube_music_search_url"])

    def test_feedback_duplicate_and_favorites_are_session_scoped(self) -> None:
        recommendation = self.client.post("/api/recommendations", headers=self.headers, json={
            "mood": "calm", "goal": "relax and unwind", "limit": 1,
        }).json["data"]["recommendations"][0]
        payload = {"recommendation_id": recommendation["recommendation_id"], "rating": "Loved it"}
        result = self.client.post("/api/feedback", headers=self.headers, json=payload)
        self.assertEqual(result.status_code, 201)
        self.assertEqual(result.json["data"]["reward"], 2.0)
        duplicate = self.client.post("/api/feedback", headers=self.headers, json=payload)
        self.assertEqual(duplicate.status_code, 409)

        saved = self.client.post("/api/favorites", headers=self.headers, json={"song_id": "song-a"})
        self.assertEqual(saved.status_code, 201)
        self.assertEqual(len(self.client.get("/api/favorites", headers=self.headers).json["data"]), 1)
        other_session = self.client.get("/api/favorites", headers={"X-Session-ID": "another-user"})
        self.assertEqual(other_session.json["data"], [])

    def test_language_filter_uses_available_metadata_and_unknown(self) -> None:
        conn = sqlite3.connect(self.database)
        conn.execute("UPDATE songs SET language='Hindi|English' WHERE song_id='song-a'")
        conn.commit(); conn.close()
        languages = self.client.get("/api/languages").json["data"]
        language_counts = {item["language"]: item["song_count"] for item in languages}
        self.assertEqual(language_counts["Hindi"], 1)
        self.assertEqual(language_counts["English"], 1)
        self.assertEqual(language_counts["Telugu"], 0)
        self.assertEqual(language_counts["Unknown"], 1)
        english = self.client.post("/api/recommendations", headers=self.headers, json={"language": "English", "limit": 2})
        self.assertEqual(english.status_code, 200)
        self.assertEqual([song["song_id"] for song in english.json["data"]["recommendations"]], ["song-a"])
        unknown = self.client.post("/api/recommendations", headers={"X-Session-ID": "unknown-user"}, json={"language": "Unknown", "limit": 2})
        self.assertEqual([song["song_id"] for song in unknown.json["data"]["recommendations"]], ["song-b"])
        no_matches = self.client.post("/api/recommendations", headers=self.headers, json={"language": "Telugu"})
        self.assertEqual(no_matches.json["data"]["recommendations"], [])
        self.assertIn("All Languages", no_matches.json["data"]["message"])

    def test_saraga_catalog_endpoints_use_real_recording_fields(self) -> None:
        health = self.client.get("/api/health").json["data"]
        self.assertEqual(health["saraga_recording_count"], 2)
        self.assertEqual({item["tradition"] for item in self.client.get("/api/traditions").json["data"]}, {"carnatic", "hindustani"})
        language_counts = {item["language"]: item["song_count"] for item in self.client.get("/api/languages").json["data"]}
        self.assertEqual(language_counts["Unknown"], 2)
        self.assertEqual(language_counts["Hindi"], 0)
        result = self.client.get("/api/recordings?raga=Raga+A").json
        self.assertEqual(result["meta"]["total"], 1)
        recording = result["data"][0]
        self.assertEqual(recording["recording_id"], "saraga-a")
        self.assertFalse(recording["audio_available"])
        self.assertNotIn("audio_path", recording)
        unavailable_language = self.client.get("/api/recordings?language=English")
        self.assertEqual(unavailable_language.status_code, 400)

    def test_saraga_recommendations_rotate_and_record_feedback(self) -> None:
        payload = {"context": {"mood": "calm", "goal": "relax", "tradition": "carnatic",
                                "preferences": {"raga": "Raga A"}}, "limit": 1}
        response = self.client.post("/api/saraga/recommendations", headers=self.headers, json=payload)
        self.assertEqual(response.status_code, 200)
        recommendation = response.json["data"]["recommendations"][0]
        self.assertEqual(recommendation["recording_id"], "saraga-a")
        self.assertIn("raga", recommendation["matched_fields"])
        duplicate_list = self.client.post("/api/saraga/recommendations", headers=self.headers, json=payload)
        self.assertTrue(duplicate_list.json["data"]["rotation_reset"])
        feedback_payload = {"recommendation_id": recommendation["recommendation_id"], "rating": "Liked it"}
        feedback = self.client.post("/api/recording-feedback", headers=self.headers, json=feedback_payload)
        self.assertEqual(feedback.status_code, 201)
        self.assertEqual(feedback.json["data"]["reward"], 1.0)
        duplicate_feedback = self.client.post("/api/recording-feedback", headers=self.headers, json=feedback_payload)
        self.assertEqual(duplicate_feedback.status_code, 409)
        saved = self.client.post("/api/recording-favorites", headers=self.headers, json={"recording_id": "saraga-a"})
        self.assertEqual(saved.status_code, 201)
        self.assertEqual(len(self.client.get("/api/recording-favorites", headers=self.headers).json["data"]), 1)
        self.assertEqual(len(self.client.get("/api/recording-history", headers=self.headers).json["data"]), 2)
        insights = self.client.get("/api/recording-insights", headers=self.headers).json["data"]
        self.assertEqual(insights["feedback_and_q_learning"]["feedback_count"], 1)
        self.assertEqual(insights["feedback_and_q_learning"]["policy_status"], "not_enabled")
        self.assertEqual(insights["selected_goals"][0]["goal"], "relax")
        self.assertEqual(self.client.get("/api/recording-insights", headers={"X-Session-ID": "other"}).json["data"]["feedback_and_q_learning"]["feedback_count"], 0)
        audio = self.client.get("/api/audio/saraga-a")
        self.assertEqual(audio.status_code, 404)

    def test_account_auth_preferences_and_private_data_scope(self) -> None:
        self.client.post("/api/favorites", headers=self.headers, json={"song_id": "song-a"})
        registered = self.client.post("/api/auth/register", headers=self.headers, json={
            "name": "Music Listener", "email": "  listener@example.com ", "password": "a-secure-password",
        })
        self.assertEqual(registered.status_code, 201)
        user = registered.json["data"]["user"]
        self.assertEqual(user["email"], "listener@example.com")
        self.assertEqual(self.client.get("/api/auth/me").json["data"]["user"]["user_id"], user["user_id"])
        self.assertEqual(len(self.client.get("/api/favorites", headers={"X-Session-ID": "test-user"}).json["data"]), 1)
        conn = sqlite3.connect(self.database)
        stored = conn.execute("SELECT password_hash FROM users WHERE user_id=?", (user["user_id"],)).fetchone()[0]
        conn.close()
        self.assertNotEqual(stored, "a-secure-password")
        self.assertTrue(stored.startswith("scrypt:"))
        saved_preferences = self.client.put("/api/preferences", json={"preferred_language": "Hindi", "favorite_genres": ["pop"]})
        self.assertEqual(saved_preferences.status_code, 200)

        second = self.app.test_client()
        self.assertEqual(second.get("/api/preferences").status_code, 401)
        self.assertEqual(second.get("/api/favorites", headers={"X-Session-ID": "test-user"}).json["data"], [])
        login = second.post("/api/auth/login", json={"email": "listener@example.com", "password": "a-secure-password"})
        self.assertEqual(login.status_code, 200)
        self.assertEqual(second.get("/api/preferences").json["data"], {"preferred_language": "Hindi", "favorite_genres": ["pop"]})
        self.assertEqual(second.get("/api/favorites").json["data"][0]["song_id"], "song-a")
        other_account = self.app.test_client()
        self.assertEqual(other_account.post("/api/auth/register", json={
            "name": "Other User", "email": "other@example.com", "password": "another-secure-password",
        }).status_code, 201)
        other_account.post("/api/favorites", json={"song_id": "song-b"})
        switch_attempt = self.client.post("/api/auth/login", json={"email": "other@example.com", "password": "another-secure-password"})
        self.assertEqual(switch_attempt.status_code, 409)
        self.assertEqual([row["song_id"] for row in self.client.get("/api/favorites").json["data"]], ["song-a"])
        self.assertEqual(self.client.post("/api/auth/register", json={"name": "Duplicate", "email": "LISTENER@example.com", "password": "a-secure-password"}).status_code, 409)
        self.assertEqual(second.post("/api/auth/logout").status_code, 200)
        self.assertIsNone(second.get("/api/auth/me").json["data"]["user"])

    def test_mood_recommendation_cards_include_cached_multilingual_lyrics_analysis(self) -> None:
        class Catalog:
            def search(self, _term, _country, _limit):
                return [{"kind": "song", "trackId": 7000 + index, "trackName": f"Track {index}",
                         "artistName": f"Performer {index}", "collectionName": f"Album {index}",
                         "trackTimeMillis": 180000, "releaseDate": "2024-01-01T00:00:00Z",
                         "primaryGenreName": "Pop", "artworkUrl100": "https://example.org/art.jpg"}
                        for index in range(60)]

        class Lyrics:
            calls = 0
            def lookup(self, song):
                self.calls += 1
                return {"status": "matched", "lyrics_available": True, "record_id": self.calls,
                        "matched_title": song["song_name"], "matched_artist": song["artist"],
                        "match_score": 1.0,
                        "lyrics_text": "I love you, my beloved. The heartbreak left tears and grief."}

        self.app.extensions["tunely_itunes_client"] = Catalog()
        provider = Lyrics()
        self.app.extensions["tunely_lyrics_client"] = provider
        response = self.client.post("/api/mood-recommendations", headers=self.headers, json={
            "language": "English", "mood": "romantic", "intensity": "Moderate", "limit": 8,
        })
        self.assertEqual(response.status_code, 200)
        recommendations = response.json["data"]["recommendations"]
        self.assertEqual(len(recommendations), 8)
        self.assertIn("Romantic", recommendations[0]["emotion_labels"])
        self.assertIn("Sad", recommendations[0]["emotion_labels"])
        self.assertEqual(recommendations[0]["lyrics_availability"], "available")
        self.assertIn("lyrics analysis", recommendations[0]["explanation"].casefold())
        self.assertEqual(provider.calls, 8)


if __name__ == "__main__":
    unittest.main()
