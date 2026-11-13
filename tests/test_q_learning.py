"""Focused Phase 3 tests for the persistent Q-learning implementation."""

import random
import tempfile
import unittest
from pathlib import Path

from backend.recommender.q_learning import (
    FeedbackConflictError,
    QLearningRecommender,
)


class QLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "tunely.sqlite3"
        self.context = {"mood": "calm", "goal": "relax and unwind", "preference": "acoustic"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_q_update_matches_standard_formula_and_persists_per_user(self) -> None:
        learner = QLearningRecommender(self.database, alpha=0.5, gamma=0.0, epsilon=0.0, rng=random.Random(1))
        result = learner.record_feedback(
            recommendation_id="recommendation-1", context=self.context,
            action="features", rating="Loved it", user_id="user-a", song_id="song-1",
        )
        self.assertEqual(result.old_q, 0.0)
        self.assertEqual(result.reward, 2.0)
        self.assertEqual(result.new_q, 1.0)

        restarted = QLearningRecommender(self.database, epsilon=0.0, rng=random.Random(1))
        chosen = restarted.choose_action(self.context, user_id="user-a")
        self.assertEqual(chosen["q_values"]["features"], 1.0)
        self.assertEqual(restarted.choose_action(self.context, user_id="user-b")["q_values"]["features"], 0.0)

    def test_duplicate_feedback_is_rejected(self) -> None:
        learner = QLearningRecommender(self.database)
        values = dict(recommendation_id="recommendation-1", context=self.context,
                      action="popularity", rating="Liked it")
        learner.record_feedback(**values)
        with self.assertRaises(FeedbackConflictError):
            learner.record_feedback(**values)

    def test_epsilon_one_explores_available_actions(self) -> None:
        learner = QLearningRecommender(self.database, epsilon=1.0, rng=random.Random(0))
        result = learner.choose_action(self.context, available_actions=("popularity",))
        self.assertEqual(result["action"], "popularity")
        self.assertTrue(result["explored"])


if __name__ == "__main__":
    unittest.main()
