"""Canonical artifact filenames and JSON helpers for local reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEMO_SESSION_META_FILENAME = "session_meta.json"
DEMO_SESSION_EVENTS_FILENAME = "events.jsonl"
DEMO_SESSION_EQUITY_FILENAME = "equity.jsonl"
DEMO_SESSION_REPORT_FILENAME = "report.json"
DEMO_SESSION_LEGACY_JOURNAL_FILENAME = "journal.json"
DEMO_SESSION_REQUIRED_FILES = (
    DEMO_SESSION_META_FILENAME,
    DEMO_SESSION_EVENTS_FILENAME,
    DEMO_SESSION_EQUITY_FILENAME,
)

STRATEGY_LEADERBOARD_FILENAME = "strategy_leaderboard.json"

LEAGUE_LEADERBOARD_FILENAME = "league_leaderboard.json"
LEAGUE_AUDIT_FILENAME = "league_audit.json"
LEAGUE_DEMO_FOLLOWUP_FILENAME = "demo_followup.json"
LEAGUE_DECISION_REVIEW_FILENAME = "decision_review.json"
LEAGUE_PROJECT_LEARNING_FILENAME = "project_learning.json"


def load_json(path: Path, *, encoding: str = "utf-8-sig") -> dict[str, Any]:
    with open(path, encoding=encoding) as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any], *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding=encoding) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path


def append_jsonl(path: Path, payload: dict[str, Any], *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding=encoding) as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: Path, *, encoding: str = "utf-8") -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding=encoding) as handle:
        for line in handle:
            current = line.strip()
            if not current:
                continue
            rows.append(json.loads(current))
    return rows


def latest_matching_file(root: Path, pattern: str) -> Path | None:
    if not root.exists():
        return None
    candidates = sorted(
        root.glob(pattern), key=lambda candidate: candidate.stat().st_mtime, reverse=True
    )
    return candidates[0] if candidates else None


def latest_dir_with_files(root: Path, required_files: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and all((path / filename).exists() for filename in required_files)
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


@dataclass
class DemoSessionArtifacts:
    session_id: str
    session_dir: Path
    meta: dict[str, Any]
    events: list[dict[str, Any]]
    equity: list[dict[str, Any]]
    report: dict[str, Any]


def load_demo_session(session_dir: Path) -> DemoSessionArtifacts:
    session_dir = Path(session_dir)
    report_path = session_dir / DEMO_SESSION_REPORT_FILENAME
    report = load_json(report_path, encoding="utf-8") if report_path.exists() else {}
    return DemoSessionArtifacts(
        session_id=session_dir.name,
        session_dir=session_dir,
        meta=load_json(session_dir / DEMO_SESSION_META_FILENAME, encoding="utf-8"),
        events=load_jsonl(session_dir / DEMO_SESSION_EVENTS_FILENAME, encoding="utf-8"),
        equity=load_jsonl(session_dir / DEMO_SESSION_EQUITY_FILENAME, encoding="utf-8"),
        report=report,
    )


def latest_demo_session(root: Path) -> DemoSessionArtifacts | None:
    session_dir = latest_dir_with_files(Path(root), DEMO_SESSION_REQUIRED_FILES)
    return load_demo_session(session_dir) if session_dir else None
