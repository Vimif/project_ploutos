from __future__ import annotations

import os

from core.artifacts import (
    DEMO_SESSION_EQUITY_FILENAME,
    DEMO_SESSION_EVENTS_FILENAME,
    DEMO_SESSION_META_FILENAME,
    DEMO_SESSION_REPORT_FILENAME,
    append_jsonl,
    latest_demo_session,
    latest_matching_file,
    load_demo_session,
    save_json,
)


def test_demo_session_helpers_round_trip(tmp_path):
    older = tmp_path / "20260404_120000"
    newer = tmp_path / "20260404_130000"

    for session_dir in (older, newer):
        save_json(
            session_dir / DEMO_SESSION_META_FILENAME,
            {"session_id": session_dir.name, "mode": "etoro"},
        )
        append_jsonl(
            session_dir / DEMO_SESSION_EVENTS_FILENAME, {"type": "signal", "ticker": "AAPL"}
        )
        append_jsonl(session_dir / DEMO_SESSION_EQUITY_FILENAME, {"equity": 100000.0})
        save_json(
            session_dir / DEMO_SESSION_REPORT_FILENAME,
            {"session_id": session_dir.name, "summary": {}},
        )

    import time

    os.utime(older, (time.time() - 100, time.time() - 100))
    os.utime(newer, (time.time(), time.time()))

    latest = latest_demo_session(tmp_path)

    assert latest is not None
    assert latest.session_id == newer.name
    assert latest.events[0]["ticker"] == "AAPL"
    assert latest.equity[0]["equity"] == 100000.0

    loaded = load_demo_session(newer)
    assert loaded.meta["mode"] == "etoro"


def test_latest_matching_file_returns_most_recent_match(tmp_path):
    older = tmp_path / "run_a" / "strategy_leaderboard.json"
    newer = tmp_path / "run_b" / "strategy_leaderboard.json"
    save_json(older, {"winner": "a"})
    save_json(newer, {"winner": "b"})
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    latest = latest_matching_file(tmp_path, "**/strategy_leaderboard.json")

    assert latest == newer
