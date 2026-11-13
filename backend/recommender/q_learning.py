"""Persistent tabular Q-Learning for Tunely recommendation strategies.

State: selected mood, listening goal, normalized broad preference, and a compact
summary of prior feedback. Actions are existing engine strategies (features and
popularity). Rewards come only from actual user ratings. No prior training data
is assumed; unseen state/action pairs start at Q=0.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Mapping, Sequence

from .ranking import GOAL_TARGETS, MOODS, RecommendationError, _normalized_text


DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "tunely.sqlite3"
ACTIONS = ("features", "popularity", "exploration")
REWARDS = {"loved it": 2.0, "liked it": 1.0, "it was okay": 0.0, "didn't like it": -1.0}
FEEDBACK_SUMMARIES = {"none", "positive", "neutral", "negative", "mixed"}


class FeedbackConflictError(ValueError):
    """Raised when feedback is submitted more than once for a recommendation."""


@dataclass(frozen=True)
class LearningUpdate:
    state_key: str
    action: str
    reward: float
    old_q: float
    new_q: float
    next_state_key: str


class QLearningRecommender:
    """Choose and learn between feature matching, legacy popularity, and exploration."""

    def __init__(
        self,
        database: str | Path = DEFAULT_DB,
        *,
        alpha: float = 0.2,
        gamma: float = 0.8,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        self.database = Path(database)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = rng or random.Random()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS q_values (
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    state_key TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('features', 'popularity', 'exploration')),
                    q_value REAL NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, state_key, action)
                );
                CREATE TABLE IF NOT EXISTS learning_feedback (
                    recommendation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'anonymous',
                    song_id TEXT,
                    state_key TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('features', 'popularity', 'exploration')),
                    rating TEXT NOT NULL CHECK(rating IN ('loved it', 'liked it', 'it was okay', 'didn''t like it')),
                    reward REAL NOT NULL,
                    next_state_key TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_learning_feedback_user_created
                    ON learning_feedback(user_id, created_at);
            """)
            for table, columns in (("q_values", "user_id, state_key, action, q_value, updated_at"),
                                   ("learning_feedback", "recommendation_id, user_id, song_id, state_key, action, rating, reward, next_state_key, created_at")):
                definition = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
                if "exploration" not in definition:
                    suffix = "_migration"
                    migration_table = table + suffix
                    conn.execute(f"DROP TABLE IF EXISTS {migration_table}")
                    if table == "q_values":
                        conn.execute(f"""CREATE TABLE {migration_table} (
                            user_id TEXT NOT NULL DEFAULT 'anonymous', state_key TEXT NOT NULL,
                            action TEXT NOT NULL CHECK(action IN ('features','popularity','exploration')),
                            q_value REAL NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
                            PRIMARY KEY (user_id, state_key, action))""")
                    else:
                        conn.execute(f"""CREATE TABLE {migration_table} (
                            recommendation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'anonymous', song_id TEXT,
                            state_key TEXT NOT NULL, action TEXT NOT NULL CHECK(action IN ('features','popularity','exploration')),
                            rating TEXT NOT NULL CHECK(rating IN ('loved it','liked it','it was okay','didn''t like it')),
                            reward REAL NOT NULL, next_state_key TEXT NOT NULL, created_at TEXT NOT NULL)""")
                    conn.execute(f"INSERT INTO {migration_table} ({columns}) SELECT {columns} FROM {table}")
                    conn.execute(f"DROP TABLE {table}")
                    conn.execute(f"ALTER TABLE {migration_table} RENAME TO {table}")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_feedback_user_created ON learning_feedback(user_id, created_at)")

    @staticmethod
    def state_key(context: Mapping[str, Any]) -> str:
        """Canonical compact state key; exact arbitrary text never enters Q state."""
        mood = _normalized_text(context.get("mood")) or "unspecified"
        goal = _normalized_text(context.get("goal")) or "unspecified"
        if mood != "unspecified" and mood not in MOODS:
            raise RecommendationError(f"Unsupported mood in learning state: {mood}.")
        if goal != "unspecified" and goal not in GOAL_TARGETS:
            raise RecommendationError(f"Unsupported goal in learning state: {goal}.")
        mood_level = context.get("mood_level", 3)
        if not isinstance(mood_level, int) or isinstance(mood_level, bool) or not 1 <= mood_level <= 5:
            raise RecommendationError("mood_level must be an integer from 1 to 5.")
        preference = _normalized_text(context.get("preference")) or "general"
        # Keep the state compact and avoid retaining arbitrary free-form preference text.
        if preference not in {"general", "acoustic", "danceable", "instrumental", "high energy", "low energy"}:
            raise RecommendationError("preference must be general, acoustic, danceable, instrumental, high energy, or low energy.")
        feedback = _normalized_text(context.get("feedback_summary")) or "none"
        if feedback not in FEEDBACK_SUMMARIES:
            raise RecommendationError(f"feedback_summary must be one of: {', '.join(sorted(FEEDBACK_SUMMARIES))}.")
        return json.dumps({"mood": mood, "mood_level": mood_level, "goal": goal, "preference": preference, "feedback": feedback}, sort_keys=True, separators=(",", ":"))

    def _q_values(self, conn: sqlite3.Connection, user_id: str, state_key: str, actions: Sequence[str]) -> dict[str, float]:
        for action in actions:
            conn.execute("INSERT OR IGNORE INTO q_values(user_id, state_key, action, q_value, updated_at) VALUES (?, ?, ?, 0, ?)",
                         (user_id, state_key, action, self._now()))
        rows = conn.execute(
            f"SELECT action, q_value FROM q_values WHERE user_id = ? AND state_key = ? AND action IN ({','.join('?' for _ in actions)})",
            (user_id, state_key, *actions),
        ).fetchall()
        return {row["action"]: float(row["q_value"]) for row in rows}

    def choose_action(self, context: Mapping[str, Any], available_actions: Sequence[str] = ACTIONS,
                      *, user_id: str = "anonymous") -> dict[str, Any]:
        """Choose with epsilon-greedy exploration; unseen Q-values are initialized to 0."""
        actions = tuple(dict.fromkeys(available_actions))
        if not actions or any(action not in ACTIONS for action in actions):
            raise RecommendationError(f"available_actions must be a non-empty subset of {ACTIONS}.")
        state = self.state_key(context)
        user_id = user_id or "anonymous"
        with self._session() as conn:
            q_values = self._q_values(conn, user_id, state, actions)
        explore = self.rng.random() < self.epsilon
        if explore:
            action = self.rng.choice(actions)
        else:
            best = max(q_values.values())
            action = self.rng.choice([candidate for candidate in actions if q_values[candidate] == best])
        return {"action": action, "state_key": state, "q_values": q_values, "explored": explore, "epsilon": self.epsilon}

    def record_feedback(
        self,
        *,
        recommendation_id: str,
        context: Mapping[str, Any],
        action: str,
        rating: str,
        next_context: Mapping[str, Any] | None = None,
        user_id: str = "anonymous",
        song_id: str | None = None,
    ) -> LearningUpdate:
        """Persist one real rating and perform one standard Q-learning update."""
        if not recommendation_id or not isinstance(recommendation_id, str):
            raise RecommendationError("recommendation_id must be a non-empty string.")
        action = _normalized_text(action)
        if action not in ACTIONS:
            raise RecommendationError(f"action must be one of: {', '.join(ACTIONS)}.")
        normalized_rating = _normalized_text(rating)
        if normalized_rating not in REWARDS:
            raise RecommendationError(f"rating must be one of: {', '.join(REWARDS)}.")
        state = self.state_key(context)
        next_state = self.state_key(next_context or context)
        reward = REWARDS[normalized_rating]
        timestamp = self._now()
        try:
            with self._session() as conn:
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute("SELECT 1 FROM learning_feedback WHERE recommendation_id = ?", (recommendation_id,)).fetchone()
                if existing:
                    raise FeedbackConflictError("Feedback was already submitted for this recommendation.")
                effective_user_id = user_id or "anonymous"
                q = self._q_values(conn, effective_user_id, state, ACTIONS)
                next_q = self._q_values(conn, effective_user_id, next_state, ACTIONS)
                old_q = q[action]
                new_q = old_q + self.alpha * (reward + self.gamma * max(next_q.values()) - old_q)
                conn.execute("""INSERT INTO learning_feedback
                    (recommendation_id, user_id, song_id, state_key, action, rating, reward, next_state_key, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (recommendation_id, effective_user_id, song_id, state, action, normalized_rating, reward, next_state, timestamp))
                conn.execute("UPDATE q_values SET q_value = ?, updated_at = ? WHERE user_id = ? AND state_key = ? AND action = ?",
                             (new_q, timestamp, effective_user_id, state, action))
                return LearningUpdate(state, action, reward, old_q, new_q, next_state)
        except sqlite3.IntegrityError as exc:
            raise FeedbackConflictError("Feedback could not be recorded because this recommendation already has feedback.") from exc

    def insights(self, *, user_id: str = "anonymous", limit: int = 20) -> dict[str, Any]:
        """Return observed ratings and learned values for one user, with small-sample caveat."""
        if not 1 <= limit <= 100:
            raise RecommendationError("limit must be between 1 and 100.")
        with self._session() as conn:
            rows = conn.execute("""SELECT action, rating, COUNT(*) AS count, AVG(reward) AS average_reward
                FROM learning_feedback WHERE user_id = ? GROUP BY action, rating ORDER BY action, rating""", (user_id or "anonymous",)).fetchall()
            q_rows = conn.execute("""SELECT state_key, action, q_value, updated_at FROM q_values
                WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?""", (user_id or "anonymous", limit)).fetchall()
            count = conn.execute("SELECT COUNT(*) FROM learning_feedback WHERE user_id = ?", (user_id or "anonymous",)).fetchone()[0]
        return {
            "feedback_count": count,
            "observations": [dict(row) for row in rows],
            "q_values": [dict(row) for row in q_rows],
            "reliability_note": "These are descriptive observations; small interaction counts are not statistically reliable.",
        }

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
