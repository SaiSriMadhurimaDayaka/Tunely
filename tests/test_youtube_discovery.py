"""Unit tests for language-aware YouTube metadata discovery."""

import io
import json
import unittest
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from backend.recommender.youtube_discovery import (
    MOOD_PROFILES,
    SUPPORTED_LANGUAGES,
    YouTubeDataAPIClient,
    YouTubeDiscoveryError,
    build_search_query,
    language_evidence,
    mood_genres,
    normalize_intensity,
    rank_candidates,
)


class YouTubeDiscoveryTests(unittest.TestCase):
    def test_all_moods_and_intensities_have_automatic_genres(self):
        for mood in MOOD_PROFILES:
            for intensity in ("mild", "moderate", "intense"):
                with self.subTest(mood=mood, intensity=intensity):
                    genres = mood_genres(mood, intensity)
                    self.assertTrue(genres)
                    self.assertTrue(all(isinstance(value, str) for value in genres))

    def test_search_query_contains_language_intensity_styles_and_optional_goal(self):
        genres = mood_genres("calm", "mild")
        query = build_search_query("Telugu", "calm", "mild", genres, goal="focus")
        self.assertIn("Telugu", query)
        self.assertIn("soft peaceful", query)
        self.assertIn("focus study", query)
        self.assertEqual(normalize_intensity("Intense"), (3, "intense"))
        self.assertEqual(normalize_intensity(4), (0, ""))

    def test_language_evidence_is_explicitly_heuristic(self):
        confidence, method = language_evidence("భావగీతం", "Telugu")
        self.assertEqual(confidence, 0.9)
        self.assertEqual(method, "writing_system_detected_in_video_text")
        weak_confidence, weak_method = language_evidence("music video", "Hindi")
        self.assertLess(weak_confidence, 0.5)
        self.assertEqual(weak_method, "search_query_only_language_unverified")
        self.assertEqual(language_evidence("song in Hindi", "Hindi")[0], 0.78)
        self.assertLess(language_evidence("English song", "Spanish")[0], 0.5)

    def test_ranker_deduplicates_and_never_invents_video_ids_or_verified_language(self):
        items = [
            {"video_id": "abcdefghijk", "title": "తెలుగు calm melody", "channel_title": "Music House", "thumbnail_url": "https://i.ytimg.com/vi/abcdefghijk/hqdefault.jpg"},
            {"video_id": "lmnopqrstuv", "title": "తెలుగు calm melody", "channel_title": "Music House"},
            {"video_id": "short", "title": "తెలుగు calm melody"},
            {"video_id": "zyxwvutsrqp", "title": "unknown language video", "channel_title": "Uploader"},
        ]
        ranked = rank_candidates(items, language="Telugu", mood="calm", intensity="mild",
                                 genres=["gentle ambient", "soft instrumental"])
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["youtube_video_id"], "abcdefghijk")
        self.assertEqual(ranked[0]["youtube_music_url"], "https://music.youtube.com/watch?v=abcdefghijk")
        self.assertFalse(ranked[0]["language_verified"])
        self.assertIsNone(ranked[0]["mood"])
        self.assertEqual(ranked[0]["selected_mood"], "calm")

    def test_youtube_api_client_builds_documented_request_and_parses_video_ids(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({"items": [{"id": {"videoId": "abcdefghijk"},
                    "snippet": {"title": "Title", "description": "Details", "channelTitle": "Channel",
                                "thumbnails": {"high": {"url": "https://i.ytimg.com/cover.jpg"}}}}]}).encode()

        captured = {}
        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return Response()

        items = YouTubeDataAPIClient("secret-key", timeout=5, opener=opener).search("Telugu calm", "te", 7)
        query = parse_qs(urlsplit(captured["url"]).query)
        self.assertEqual(query["type"], ["video"])
        self.assertEqual(query["relevanceLanguage"], ["te"])
        self.assertEqual(query["maxResults"], ["7"])
        self.assertNotIn("secret-key", repr(items))
        self.assertEqual(items[0]["video_id"], "abcdefghijk")
        self.assertEqual(items[0]["thumbnail_url"], "https://i.ytimg.com/cover.jpg")

    def test_quota_failure_has_a_specific_error(self):
        body = io.BytesIO(json.dumps({"error": {"errors": [{"reason": "quotaExceeded"}]}}).encode())
        def opener(*args, **kwargs):
            raise HTTPError("https://example.test", 403, "Forbidden", {}, body)
        with self.assertRaises(YouTubeDiscoveryError) as raised:
            YouTubeDataAPIClient("key", opener=opener).search("query", "te")
        self.assertEqual(raised.exception.code, "quota_exceeded")

    def test_network_failure_has_a_search_fallback_error_code(self):
        def opener(*args, **kwargs):
            raise URLError("offline")
        with self.assertRaises(YouTubeDiscoveryError) as raised:
            YouTubeDataAPIClient("key", opener=opener).search("query", "te")
        self.assertEqual(raised.exception.code, "network_error")


if __name__ == "__main__":
    unittest.main()
