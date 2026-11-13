"""Evidence-based multi-label music category classification tests."""

import unittest
from datetime import date

from backend.recommender.music_categories import CATEGORY_IDS, classify_music_categories, infer_music_categories


class MusicCategoryTests(unittest.TestCase):
    TODAY = date(2026, 11, 9)

    def labels(self, item, features=None):
        return classify_music_categories(item, audio_features=features, today=self.TODAY)

    def test_all_seven_categories_use_their_supported_evidence(self):
        examples = {
            "mass": ({"primaryGenreName": "Mass"}, None),
            "party": ({"primaryGenreName": "Dance"}, None),
            "slow": ({"primaryGenreName": "Ballad"}, None),
            "old": ({"releaseDate": "1990-01-01T00:00:00Z"}, None),
            "latest": ({"releaseDate": "2025-04-01T00:00:00Z"}, None),
            "romantic": ({"trackName": "Love Forever", "collectionName": "Romantic Collection"}, None),
            "sad": ({"trackName": "Tears of Pain", "collectionName": "Sad Heartbreak"}, None),
        }
        for category, (item, features) in examples.items():
            with self.subTest(category=category):
                self.assertIn(category, self.labels(item, features)["category_ids"])

    def test_multiple_categories_can_match_one_song(self):
        result = self.labels({"primaryGenreName": "Mass Dance", "releaseDate": "1990-01-01"})
        self.assertIn("mass", result["category_ids"])
        self.assertIn("party", result["category_ids"])
        self.assertIn("old", result["category_ids"])

    def test_audio_categories_require_combined_features(self):
        self.assertNotIn("party", self.labels({"trackName": "Dance Party Hit"})["category_ids"])
        features = {"energy": 0.9, "danceability": 0.85, "tempo": 125}
        self.assertIn("party", self.labels({}, features)["category_ids"])
        self.assertIn("mass", self.labels({"collectionName": "Mass Film Album", "primaryGenreName": "Indian"}, features)["category_ids"])
        self.assertIn("slow", self.labels({}, {"tempo": 72, "energy": 0.3})["category_ids"])

    def test_title_alone_does_not_claim_romantic_or_sad(self):
        result = self.labels({"trackName": "Love and Tears"})
        self.assertNotIn("romantic", result["category_ids"])
        self.assertNotIn("sad", result["category_ids"])

    def test_emotional_labels_are_not_inferred_from_provider_genre_alone(self):
        result = self.labels({"primaryGenreName": "Romantic"})
        self.assertNotIn("romantic", result["category_ids"])
        result = self.labels({"primaryGenreName": "Sad"})
        self.assertNotIn("sad", result["category_ids"])

    def test_missing_invalid_or_future_dates_do_not_claim_old_or_latest(self):
        for item in ({}, {"releaseDate": "unknown"}, {"releaseDate": "2030-01-01"}):
            result = self.labels(item)
            self.assertNotIn("old", result["category_ids"])
            self.assertNotIn("latest", result["category_ids"])

    def test_every_mood_and_listening_goal_combination_infers_supported_preferences(self):
        moods = ("happy", "sad", "calm", "energetic", "stressed", "romantic", "neutral", "motivated", "nostalgic")
        goals = ("relax", "focus", "energize", "feel-good", "wind down", "discover something new")
        for mood in moods:
            for goal in goals:
                with self.subTest(mood=mood, goal=goal):
                    inferred = infer_music_categories(mood, goal)
                    self.assertLessEqual(set(inferred), CATEGORY_IDS)
        self.assertEqual(infer_music_categories("happy", "energize")[:2], ["party", "mass"])
        self.assertEqual(infer_music_categories("energetic", "energize")[:2], ["party", "mass"])
        self.assertIn("slow", infer_music_categories("calm", "relax"))
        self.assertIn("romantic", infer_music_categories("romantic", "wind down"))
        self.assertIn("sad", infer_music_categories("sad", "wind down"))
        self.assertIn("old", infer_music_categories("nostalgic", "discover something new"))


if __name__ == "__main__":
    unittest.main()
