"""Checks for the three-level, feature-based mood intensity controls."""

import unittest

from backend.recommender.ranking import RecommendationError, targets_for_context


class MoodIntensityTests(unittest.TestCase):
    def test_intensity_scales_mood_targets(self):
        mild = targets_for_context({"mood": "happy", "mood_level": 1})
        moderate = targets_for_context({"mood": "happy", "mood_level": 2})
        intense = targets_for_context({"mood": "happy", "mood_level": 3})
        self.assertLess(mild["energy"], moderate["energy"])
        self.assertLess(moderate["energy"], intense["energy"])
        self.assertLess(mild["valence"], moderate["valence"])
        self.assertLess(moderate["valence"], intense["valence"])

    def test_listening_goal_remains_blended_with_mood(self):
        result = targets_for_context({"mood": "sad", "mood_level": 2, "goal": "energize"})
        self.assertIn("energy", result)
        self.assertIn("danceability", result)
        self.assertGreater(result["energy"], 0.3)

    def test_rejects_levels_outside_three_level_scale(self):
        for value in (0, 4, "intense", True):
            with self.subTest(value=value), self.assertRaises(RecommendationError):
                targets_for_context({"mood": "calm", "mood_level": value})


if __name__ == "__main__":
    unittest.main()
