"""Descriptive, feedback-based evaluation for Tunely's logged recommendations.

This module reads real recommendation and feedback records only. It reports
observed outcomes for the strategy selected by the running Q-Learning policy;
it does not estimate counterfactual outcomes for strategies that were not used.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import Any

from .ranking import DEFAULT_DB


STRATEGIES = ("popularity", "features", "exploration")


def _rates(rows: list[dict[str, Any]], catalog_size: int) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    distinct_songs = len({row["song_id"] for row in rows})
    count = len(rows)
    return {
        "feedback_count": count,
        "positive_feedback_count": sum(reward > 0 for reward in rewards),
        "acceptance_rate": round(sum(reward > 0 for reward in rewards) / count, 4) if count else None,
        "average_reward": round(sum(rewards) / count, 4) if count else None,
        "catalog_coverage_count": distinct_songs,
        "catalog_coverage_rate": round(distinct_songs / catalog_size, 4) if catalog_size else None,
    }


def evaluate(database: str | Path = DEFAULT_DB, *, user_id: str | None = None) -> dict[str, Any]:
    """Summarize logged outcomes by strategy and listening goal.

    Acceptance is the share of submitted ratings with positive rewards. Catalog
    coverage uses songs with submitted feedback, divided by the catalog size.
    A user filter can scope the report to one anonymous session.
    """
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(f"Tunely database not found: {path}")

    try:
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "songs" not in tables:
                raise ValueError("The database has no songs table. Prepare the catalog first.")
            catalog_size = int(conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0])
            if "recommendations" not in tables or "learning_feedback" not in tables:
                return _empty_report(path, catalog_size, user_id)

            where = " WHERE r.user_id = ?" if user_id else ""
            params = (user_id,) if user_id else ()
            recommendation_count = int(conn.execute(
                f"SELECT COUNT(*) FROM recommendations r{where}", params
            ).fetchone()[0])
            rows = [dict(row) for row in conn.execute(f"""
                SELECT r.recommendation_id, r.user_id, r.song_id, r.context_json,
                       r.strategy, r.created_at, f.rating, f.reward
                FROM recommendations r
                LEFT JOIN learning_feedback f ON f.recommendation_id = r.recommendation_id
                {where}
                ORDER BY r.created_at
            """, params)]
    except sqlite3.Error as exc:
        raise ValueError("Could not read Tunely's evaluation data.") from exc

    rated_by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rated_by_goal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        strategy = row["strategy"]
        all_by_strategy[strategy].append(row)
        try:
            context = json.loads(row["context_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            context = {}
        row["goal"] = str(context.get("goal") or "unspecified")
        if row["reward"] is not None:
            rated_by_strategy[strategy].append(row)
            rated_by_goal[row["goal"]].append(row)

    strategy_metrics: dict[str, Any] = {}
    for strategy in (*STRATEGIES, "other"):
        matching = rated_by_strategy.get(strategy, [])
        strategy_metrics[strategy] = {
            **_rates(matching, catalog_size),
            "recommendation_count": len(all_by_strategy.get(strategy, [])),
        }

    goal_metrics = {
        goal: _rates(goal_rows, catalog_size)
        for goal, goal_rows in sorted(rated_by_goal.items())
    }
    feedback_count = sum(row["reward"] is not None for row in rows)
    rated_rows = [row for row in rows if row["reward"] is not None]
    return {
        "status": "ok" if feedback_count else "insufficient_data",
        "data_source": "real_user_feedback",
        "user_id": user_id,
        "catalog_size": catalog_size,
        "recommendation_count": recommendation_count,
        "feedback_count": feedback_count,
        "overall": _rates(rated_rows, catalog_size),
        "observed_strategy_outcomes": strategy_metrics,
        "by_listening_goal": goal_metrics,
        "personalization_evaluation": {
            "policy": "q_learning_selected_strategy",
            "recommendation_count": recommendation_count,
            "feedback_count": feedback_count,
            "interpretation": (
                "These are outcomes for strategies selected by the logged Q-Learning policy. "
                "They are observational and do not establish that Q-Learning caused better outcomes."
            ),
        },
        "limitations": [
            "Only recommendations with submitted ratings contribute to feedback metrics.",
            "The log records the selected strategy, not outcomes for unselected counterfactual strategies.",
            "Small samples are descriptive and are not statistically reliable.",
            "Preference prediction is not evidence that music changes a listener's emotional state.",
        ],
    }


def _empty_report(path: Path, catalog_size: int, user_id: str | None) -> dict[str, Any]:
    return {
        "status": "insufficient_data",
        "data_source": "real_user_feedback",
        "database": str(path),
        "user_id": user_id,
        "catalog_size": catalog_size,
        "recommendation_count": 0,
        "feedback_count": 0,
        "overall": _rates([], catalog_size),
        "observed_strategy_outcomes": {
            strategy: {**_rates([], catalog_size), "recommendation_count": 0}
            for strategy in (*STRATEGIES, "other")
        },
        "by_listening_goal": {},
        "personalization_evaluation": {
            "policy": "q_learning_selected_strategy",
            "recommendation_count": 0,
            "feedback_count": 0,
            "interpretation": "No logged Q-Learning outcomes are available yet.",
        },
        "limitations": ["No rated recommendation records are available; no simulated results were substituted."],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Tunely's real logged recommendation feedback.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--user-id", help="Optionally limit results to one X-Session-ID.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this file instead of stdout.")
    args = parser.parse_args()
    report = evaluate(args.database, user_id=args.user_id)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Evaluation report saved to {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
