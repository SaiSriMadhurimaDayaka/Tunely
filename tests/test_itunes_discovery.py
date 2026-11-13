"""Tests for no-key iTunes song discovery and conservative metadata ranking."""

import io
import json
import ssl
import unittest
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

from backend.recommender.itunes_discovery import (
    ITunesDiscoveryError,
    ITunesSearchClient,
    build_itunes_term,
    build_itunes_query_variants,
    build_itunes_seed_query,
    rank_itunes_results,
)


class ITunesDiscoveryTests(unittest.TestCase):
    def test_documented_search_request_and_metadata_parsing(self):
        captured = {}

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self):
                return json.dumps({"results": [
                    {"kind": "song", "trackId": 123, "trackName": "తెలుగు పాట", "artistName": "Singer",
                     "primaryGenreName": "Indian", "artworkUrl100": "http://is1-ssl.mzstatic.com/cover.jpg"},
                    {"kind": "music-video", "trackId": 456, "trackName": "Not a song"},
                ]}).encode()

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            return Response()

        results = ITunesSearchClient(timeout=5, opener=opener).search("Telugu calm song", "in", 50)
        query = parse_qs(urlsplit(captured["url"]).query)
        self.assertEqual(query["entity"], ["song"])
        self.assertEqual(query["country"], ["in"])
        self.assertEqual(query["term"], ["Telugu calm song"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["trackId"], 123)

    def test_search_targets_selected_language_and_generates_mood_genres_automatically(self):
        term, genres = build_itunes_term("Telugu", "calm", "Intense", "relax")
        self.assertEqual(term, "Telugu songs")
        self.assertTrue(genres)
        self.assertIn("ambient", " ".join(genres))

    def test_category_search_terms_preserve_selected_language(self):
        term, _ = build_itunes_term("Telugu", "happy", "Moderate", category="party")
        self.assertEqual(term, "Telugu party dance songs")

    def test_queries_change_with_mood_intensity_goal_and_category_preferences(self):
        happy = build_itunes_query_variants("Telugu", "happy", "Mild", "feel-good", ["party"])
        energetic = build_itunes_query_variants("Telugu", "energetic", "Intense", "energize", ["mass", "party"])
        calm = build_itunes_query_variants("Telugu", "calm", "Moderate", "relax", ["slow"])
        self.assertNotEqual(happy, energetic)
        self.assertNotEqual(energetic, calm)
        self.assertTrue(any("party" in query.casefold() for query in happy))
        self.assertTrue(any("mass" in query.casefold() for query in energetic))
        self.assertTrue(any("relax" in query.casefold() or "slow" in query.casefold() for query in calm))

    def test_seed_query_uses_real_provider_artist_or_album_metadata(self):
        items = [{"artistName": "Singer A", "collectionName": "Album A"}]
        self.assertEqual(build_itunes_seed_query("Telugu", items), "Telugu Singer A")
        self.assertEqual(build_itunes_seed_query("Telugu", items, recent_artists=["Singer A"]), "Telugu Album A")
        self.assertIsNone(build_itunes_seed_query("Telugu", items, recent_artists=["Singer A"], recent_albums=["Album A"]))

    def test_ranker_keeps_tracks_without_optional_language_metadata_as_unknown(self):
        items = [
            {"kind": "song", "trackId": 1, "trackName": "తెలుగు calm పాట", "artistName": "Singer",
             "collectionName": "Album", "primaryGenreName": "Ambient"},
            {"kind": "song", "trackId": 2, "trackName": "Latin title", "artistName": "Singer",
             "collectionName": "Album", "primaryGenreName": "Pop"},
        ]
        results = rank_itunes_results(items, language="Telugu", mood="calm", intensity="Mild",
                                      genres=["gentle ambient", "ambient"], limit=8)
        self.assertEqual({item["itunes_track_id"] for item in results}, {"1", "2"})
        self.assertFalse(results[0]["language_verified"])
        unknown = next(item for item in results if item["itunes_track_id"] == "2")
        self.assertEqual(unknown["language"], "Unknown")
        self.assertEqual(unknown["requested_language"], "Telugu")
        self.assertIn("does not verify the track language", unknown["explanation"])
        self.assertIsNone(results[0].get("youtube_video_id"))
        self.assertEqual(parse_qs(urlsplit(results[0]["youtube_music_search_url"]).query)["q"], ["తెలుగు calm పాట Singer"])

    def test_rate_limit_and_network_failures_are_actionable(self):
        def rate_limited(*args, **kwargs):
            raise HTTPError("https://itunes.apple.com/search", 429, "Too Many Requests", {}, io.BytesIO())
        with self.assertRaises(ITunesDiscoveryError) as raised:
            ITunesSearchClient(opener=rate_limited).search("query")
        self.assertEqual(raised.exception.code, "rate_limited")

        def offline(*args, **kwargs):
            raise URLError("offline")
        with self.assertRaises(ITunesDiscoveryError) as raised:
            ITunesSearchClient(opener=offline).search("query")
        self.assertEqual(raised.exception.code, "network_error")

    def test_network_restriction_timeout_and_tls_are_distinguished(self):
        cases = [
            (lambda *args, **kwargs: (_ for _ in ()).throw(URLError(PermissionError(10013, "socket access denied"))),
             "network_restricted"),
            (lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")), "timeout"),
            (lambda *args, **kwargs: (_ for _ in ()).throw(ssl.SSLError("CERTIFICATE_VERIFY_FAILED")), "tls_error"),
        ]
        for opener, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(ITunesDiscoveryError) as raised:
                    ITunesSearchClient(timeout=3, opener=opener).search("Telugu calm")
                self.assertEqual(raised.exception.code, expected_code)
                self.assertIn("iTunes", str(raised.exception))

    def test_invalid_response_and_empty_results(self):
        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return self.body

        for body in (b"not json", json.dumps({"resultCount": 1}).encode()):
            with self.subTest(body=body):
                with self.assertRaises(ITunesDiscoveryError) as raised:
                    ITunesSearchClient(opener=lambda *args, _body=body, **kwargs: Response(_body)).search("query")
                self.assertEqual(raised.exception.code, "invalid_response")

        empty = ITunesSearchClient(opener=lambda *args, **kwargs: Response(b'{"resultCount":0,"results":[]}')).search("query")
        self.assertEqual(empty, [])

    def test_ranker_filters_by_evidence_and_returns_multiple_labels(self):
        items = [
            {"kind": "song", "trackId": 10, "trackName": "One", "primaryGenreName": "Dance"},
            {"kind": "song", "trackId": 11, "trackName": "Two", "primaryGenreName": "Classical"},
        ]
        results = rank_itunes_results(items, language="Telugu", mood="happy", intensity="Moderate",
                                      genres=["dance"], category="party")
        self.assertEqual([item["itunes_track_id"] for item in results], ["10"])
        self.assertIn("Party Songs", results[0]["music_categories"])

    def test_ranker_deduplicates_and_prefers_unseen_tracks_and_distinct_artists(self):
        items = [
            {"kind": "song", "trackId": 1, "trackName": "One", "artistName": "Shared", "collectionName": "A"},
            {"kind": "song", "trackId": 1, "trackName": "One duplicate", "artistName": "Shared", "collectionName": "A"},
            {"kind": "song", "trackId": 2, "trackName": "Two", "artistName": "Shared", "collectionName": "A"},
            *[{"kind": "song", "trackId": index, "trackName": f"Track {index}",
               "artistName": f"Artist {index}", "collectionName": f"Album {index}"}
              for index in range(3, 21)],
        ]
        kwargs = {"language": "Telugu", "mood": "happy", "intensity": "Moderate", "genres": ["upbeat pop"], "limit": 8}
        initial = rank_itunes_results(items, **kwargs)
        self.assertEqual(len({row["itunes_track_id"] for row in initial}), len(initial))
        self.assertEqual(len({row["artist"] for row in initial}), len(initial))
        next_list = rank_itunes_results(items, **kwargs, recent_track_ids=[row["itunes_track_id"] for row in initial])
        self.assertFalse(set(row["itunes_track_id"] for row in initial) & set(row["itunes_track_id"] for row in next_list))
        limited = rank_itunes_results(items[:2], **kwargs, recent_track_ids=["1"])
        self.assertTrue(limited)  # Reuses only when there are no fresh alternatives.

    def test_mood_intensity_and_listening_goal_change_relevance_order(self):
        items = [
            {"kind": "song", "trackId": 21, "trackName": "Happy light feel good", "artistName": "A", "primaryGenreName": "Dance"},
            {"kind": "song", "trackId": 22, "trackName": "Celebration energetic powerful", "artistName": "B", "primaryGenreName": "Dance"},
            {"kind": "song", "trackId": 23, "trackName": "Calm soothing relax and unwind", "artistName": "C", "primaryGenreName": "Ambient"},
            {"kind": "song", "trackId": 24, "trackName": "Sad heartbreak emotional wind down", "artistName": "D", "primaryGenreName": "Ballad"},
        ]
        genres_happy = ["upbeat pop", "dance", "cheerful acoustic"]
        mild = rank_itunes_results(items, language="Telugu", mood="happy", intensity="Mild",
                                   genres=genres_happy, goal="feel-good", limit=4)
        intense = rank_itunes_results(items, language="Telugu", mood="happy", intensity="Intense",
                                      genres=genres_happy, goal="energize", limit=4)
        calm = rank_itunes_results(items, language="Telugu", mood="calm", intensity="Mild",
                                   genres=["ambient", "soft instrumental"], goal="relax", limit=4)
        sad = rank_itunes_results(items, language="Telugu", mood="sad", intensity="Moderate",
                                  genres=["emotional melody", "ballad"], goal="wind down", limit=4)
        self.assertEqual(mild[0]["itunes_track_id"], "21")
        self.assertEqual(intense[0]["itunes_track_id"], "22")
        self.assertEqual(calm[0]["itunes_track_id"], "23")
        self.assertEqual(sad[0]["itunes_track_id"], "24")

    def test_nostalgic_recommendations_omit_latest_releases_unless_new_music_is_requested(self):
        items = [
            {"kind": "song", "trackId": 31, "trackName": "Recent Track", "artistName": "New Artist",
             "primaryGenreName": "Pop", "releaseDate": "2026-02-24T00:00:00Z"},
            {"kind": "song", "trackId": 32, "trackName": "Classic Track", "artistName": "Classic Artist",
             "primaryGenreName": "Pop", "releaseDate": "1990-02-24T00:00:00Z"},
        ]
        kwargs = {"language": "Telugu", "mood": "nostalgic", "intensity": "Moderate",
                  "genres": ["classic melodies"], "limit": 8}
        ordinary = rank_itunes_results(items, goal="wind down", **kwargs)
        self.assertIn("32", {row["itunes_track_id"] for row in ordinary})
        self.assertNotIn("31", {row["itunes_track_id"] for row in ordinary})
        latest_only = rank_itunes_results(items[:1], goal="wind down", **kwargs)
        self.assertEqual(latest_only, [])
        explicit_new_music = rank_itunes_results(items[:1], goal="discover something new", **kwargs)
        self.assertEqual([row["itunes_track_id"] for row in explicit_new_music], ["31"])


if __name__ == "__main__":
    unittest.main()
