import unittest

from backend.recommender.intent_ml import DEFAULT_MODEL_PATH, INTENSITIES, MOODS, load_intent_model
from backend.recommender.reviewed_catalog import APPROVED_GENRES


class IntentModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_intent_model(DEFAULT_MODEL_PATH)

    def test_saved_model_loads_and_scores_all_approved_genres(self):
        self.assertIsNotNone(self.model)
        scores = self.model.predict_scores("devotional prayer songs", "neutral", "moderate")
        self.assertEqual(tuple(scores), APPROVED_GENRES)
        self.assertTrue(all(0 <= value <= 1 for value in scores.values()))
        self.assertGreater(scores["Devotional"], scores["Sad"])

    def test_free_text_changes_genre_relevance(self):
        devotional = self.model.predict_scores("devotional prayer songs", "neutral", "moderate")
        heartbreak = self.model.predict_scores("heartbreak and separation songs", "neutral", "moderate")
        self.assertGreater(devotional["Devotional"], devotional["Sad"])
        self.assertGreater(heartbreak["Sad"], heartbreak["Devotional"])

    def test_all_mood_and_intensity_encodings_are_supported(self):
        for mood in MOODS:
            for intensity in INTENSITIES:
                with self.subTest(mood=mood, intensity=intensity):
                    scores = self.model.predict_scores("music for this moment", mood, intensity)
                    self.assertEqual(set(scores), set(APPROVED_GENRES))


if __name__ == "__main__":
    unittest.main()
