"""Tests for conservative six-language lyric emotion classification and LRCLIB."""

import json
import sqlite3
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from backend.recommender.emotion_analysis import EMOTION_LABELS, _CUES, classify_lyrics
from backend.recommender.emotion_pipeline import enrich_song_emotion
from backend.recommender.itunes_dataset import ensure_itunes_dataset_schema, upsert_itunes_tracks
from backend.recommender.lyrics_provider import LRCLibClient, select_verified_lyrics
from backend.recommender.lyrics_provider import LyricsProviderError


class EmotionPipelineTests(unittest.TestCase):
    def test_all_six_languages_process_native_script_or_romanized_cues(self):
        languages = ("Telugu", "Hindi", "Tamil", "Malayalam", "Kannada", "English")
        for language in languages:
            with self.subTest(language=language):
                for label in EMOTION_LABELS:
                    with self.subTest(label=label):
                        cues = _CUES[language][label]
                        result = classify_lyrics(f"{cues[0]} ... {cues[1]}", language)
                        self.assertIn(label, result["labels"])
                        self.assertEqual(result["status"], "classified")
                        self.assertEqual(result["model_version"], "multilingual-lyrics-lexicon-v1")

    def test_sad_romantic_overlap_and_insufficient_evidence(self):
        result = classify_lyrics("I love you, my beloved. The heartbreak left tears and grief.", "English")
        self.assertIn("Romantic", result["labels"])
        self.assertIn("Sad", result["labels"])
        self.assertEqual(classify_lyrics("slow song", "English")["status"], "unknown")
        self.assertEqual(classify_lyrics("title says sad and romantic", "English")["status"], "unknown")

    def test_slow_is_not_a_lyric_emotion(self):
        result = classify_lyrics("lonely tears, sorrow and heartbreak", "English")
        self.assertNotIn("Slow", result["labels"])
        self.assertTrue(set(result["labels"]).issubset(set(EMOTION_LABELS)))

    def test_lrclib_title_artist_match_and_alternate_spelling_validation(self):
        item = {"trackName": "Kaadhal Enbadhu", "artistName": "Singer A", "collectionName": "Album", "trackTimeMillis": 180000}
        records = [
            {"trackName": "Wrong Song", "artistName": "Singer A", "duration": 180, "plainLyrics": "bad"},
            {"trackName": "Kadhal Enbadhu", "artistName": "Singer A", "duration": 181, "plainLyrics": "love you"},
        ]
        chosen = select_verified_lyrics(records, item)
        self.assertEqual(chosen["trackName"], "Kadhal Enbadhu")
        self.assertIsNone(select_verified_lyrics([
            {"trackName": "Kaadhal Enbadhu", "artistName": "Other Artist", "duration": 180, "plainLyrics": "wrong"}
        ], item))
        self.assertIsNone(select_verified_lyrics([
            {"trackName": "Kaadhal Enbadhu", "artistName": "Singer A", "duration": 220, "plainLyrics": "wrong version"}
        ], item))

    def test_lrclib_client_sends_title_and_artist_and_returns_only_validated_match(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps([{"id": 12, "trackName": "Song (Live)", "artistName": "Artist",
                "duration": 120, "plainLyrics": "real lyric text"}]).encode()

        seen = {}
        def opener(request, timeout):
            seen["url"] = request.full_url
            seen["agent"] = request.get_header("User-agent")
            seen["timeout"] = timeout
            return Response()

        result = LRCLibClient(opener=opener).lookup({"title": "Song", "artist": "Artist", "duration_ms": 120000})
        self.assertTrue(result["lyrics_available"])
        self.assertEqual(result["lyrics_text"], "real lyric text")
        query = parse_qs(urlsplit(seen["url"]).query)
        self.assertEqual(query["track_name"], ["Song"])
        self.assertEqual(query["artist_name"], ["Artist"])
        self.assertIn("Tunely/", seen["agent"])

    def test_lrclib_rate_limit_honors_retry_after_without_retrying_early(self):
        calls = []
        headers = Message()
        headers["Retry-After"] = "7"
        def opener(*_args, **_kwargs):
            calls.append(True)
            raise HTTPError("https://lrclib.net/api/search", 429, "Too Many Requests", headers, None)
        client = LRCLibClient(opener=opener)
        with self.assertRaises(LyricsProviderError) as first:
            client.lookup({"title": "Track", "artist": "Artist"})
        self.assertEqual(first.exception.code, "lyrics_rate_limited")
        self.assertEqual(first.exception.retry_after, 7)
        with self.assertRaises(LyricsProviderError) as second:
            client.lookup({"title": "Another Track", "artist": "Artist"})
        self.assertEqual(second.exception.code, "lyrics_rate_limited")
        self.assertEqual(len(calls), 1)

    def test_incremental_cache_persists_analysis_but_never_lyrics(self):
        with tempfile.TemporaryDirectory() as temp:
            db = sqlite3.connect(Path(temp) / "emotion.sqlite3")
            db.row_factory = sqlite3.Row
            ensure_itunes_dataset_schema(db)
            upsert_itunes_tracks(db, [{"kind": "song", "trackId": 123, "trackName": "Love Song",
                "artistName": "Singer", "trackTimeMillis": 180000}], requested_language="English", country="in")

            class Provider:
                calls = 0
                def lookup(self, _song):
                    self.calls += 1
                    return {"status": "matched", "lyrics_available": True, "record_id": 22,
                            "matched_title": "Love Song", "matched_artist": "Singer", "match_score": 1.0,
                            "lyrics_text": "the lyric text must not be persisted"}

            class Classifier:
                def classify(self, _lyrics, _language):
                    return {"labels": ["Romantic"], "confidence": {"Romantic": .7},
                            "evidence": {"Romantic": ["love", "beloved"]}, "status": "classified",
                            "model_version": "test-v1", "confidence_kind": "test"}

            provider = Provider()
            song = {"itunes_track_id": "123", "song_name": "Love Song", "artist": "Singer", "duration_ms": 180000}
            enrich_song_emotion(db, song, "English", provider=provider, classifier=Classifier())
            enrich_song_emotion(db, song, "English", provider=provider, classifier=Classifier())
            row = db.execute("SELECT lyrics_availability,emotion_labels_json,emotion_model_version FROM itunes_tracks WHERE track_id='123'").fetchone()
            self.assertEqual(row["lyrics_availability"], "available")
            self.assertEqual(json.loads(row["emotion_labels_json"]), ["Romantic"])
            self.assertEqual(row["emotion_model_version"], "test-v1")
            self.assertEqual(provider.calls, 1)
            cache = dict(db.execute("SELECT * FROM itunes_lyrics_cache").fetchone())
            self.assertNotIn("lyrics_text", cache)
            self.assertNotIn("the lyric text must not be persisted", json.dumps(cache))
            db.close()

    def test_provider_failure_is_cached_without_breaking_recommendation(self):
        db = sqlite3.connect(":memory:")
        db.row_factory = sqlite3.Row
        ensure_itunes_dataset_schema(db)

        class Offline:
            calls = 0
            def lookup(self, _song):
                self.calls += 1
                raise LyricsProviderError("offline", "lyrics_network_error")

        provider = Offline()
        song = {"itunes_track_id": "999", "song_name": "Example", "artist": "Artist"}
        result = enrich_song_emotion(db, song, "Hindi", provider=provider)
        enrich_song_emotion(db, song, "Hindi", provider=provider)
        self.assertEqual(result["emotion_status"], "unknown")
        self.assertEqual(result["lyrics_lookup_status"], "lyrics_network_error")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(db.execute("SELECT COUNT(*) FROM itunes_lyrics_cache").fetchone()[0], 1)
        db.close()


if __name__ == "__main__":
    unittest.main()
