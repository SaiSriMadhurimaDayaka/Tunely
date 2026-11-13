"""Idempotently load data/telugu_songs_reviewed.xlsx into Tunely's SQLite DB."""

from __future__ import annotations

import argparse
import sqlite3

from backend.recommender.ranking import DEFAULT_DB
from backend.recommender.reviewed_catalog import (
    DEFAULT_REVIEWED_WORKBOOK, ensure_reviewed_catalog_schema, import_reviewed_workbook,
    reviewed_catalog_statistics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", default=str(DEFAULT_REVIEWED_WORKBOOK))
    parser.add_argument("--database", default=str(DEFAULT_DB))
    args = parser.parse_args()
    conn = sqlite3.connect(args.database)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        ensure_reviewed_catalog_schema(conn)
        with conn:
            report = import_reviewed_workbook(conn, args.workbook)
        report["language_counts"] = reviewed_catalog_statistics(conn)["language_counts"]
        report["genre_counts"] = reviewed_catalog_statistics(conn)["genre_counts"]
        print(report)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
