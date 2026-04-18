"""Read-only monitoring helpers for the eToro demo dashboard."""

from __future__ import annotations

import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from core.artifacts import (
    LEAGUE_DECISION_REVIEW_FILENAME,
    LEAGUE_LEADERBOARD_FILENAME,
    LEAGUE_PROJECT_LEARNING_FILENAME,
    STRATEGY_LEADERBOARD_FILENAME,
    DemoSessionArtifacts,
    latest_demo_session,
    latest_matching_file,
    load_json,
)
from trading.live_execution import create_live_broker_adapter


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


class SessionStore:
    """Load canonical paper trading sessions from disk."""

    def __init__(self, paper_trading_dir: Path):
        self.paper_trading_dir = Path(paper_trading_dir)

    def latest(self) -> DemoSessionArtifacts | None:
        return latest_demo_session(self.paper_trading_dir)


class BrokerSnapshotCache:
    """Short-lived cache around the live broker adapter."""

    def __init__(self, ttl_seconds: int = 15):
        self.ttl_seconds = int(ttl_seconds)
        self._adapter = None
        self._mode = None
        self._snapshot: dict[str, Any] | None = None
        self._expires_at = 0.0

    def get_snapshot(self, *, mode: str, initial_balance: float) -> dict[str, Any]:
        now = time.time()
        if self._snapshot and self._mode == mode and now < self._expires_at:
            return self._snapshot

        try:
            if self._adapter is None or self._mode != mode:
                self._adapter = create_live_broker_adapter(
                    mode,
                    initial_balance=initial_balance,
                    fill_timeout=30,
                    poll_interval=1.0,
                )
                self._mode = mode
            account = self._adapter.get_account()
            positions = list(self._adapter.get_positions_map().values())
            open_orders = list(self._adapter.get_open_orders())
            snapshot = {
                "connected": True,
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
                "account": account,
                "positions": positions,
                "open_orders": open_orders,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - exercised via integration mocks
            snapshot = {
                "connected": False,
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
                "account": {},
                "positions": [],
                "open_orders": [],
                "error": str(exc),
            }
            self._adapter = None

        self._snapshot = snapshot
        self._expires_at = now + self.ttl_seconds
        return snapshot


class DemoMonitorService:
    """Aggregate live demo state, telemetry, and improvement insights."""

    def __init__(
        self,
        *,
        paper_trading_dir: str | Path = "logs/paper_trading",
        strategy_compare_dir: str | Path = "logs/strategy_compare",
        profitability_dir: str | Path = "logs/profitability",
        league_batches_dir: str | Path = "logs/league_batches",
        cache_ttl_seconds: int = 15,
    ):
        self.sessions = SessionStore(Path(paper_trading_dir))
        self.strategy_compare_dir = Path(strategy_compare_dir)
        self.profitability_dir = Path(profitability_dir)
        self.league_batches_dir = Path(league_batches_dir)
        self.broker_cache = BrokerSnapshotCache(ttl_seconds=cache_ttl_seconds)

    def _latest_leaderboard(self) -> dict[str, Any] | None:
        path = latest_matching_file(
            self.strategy_compare_dir, f"**/{STRATEGY_LEADERBOARD_FILENAME}"
        )
        if not path:
            return None
        payload = load_json(path, encoding="utf-8")
        payload["_path"] = str(path)
        return payload

    def _latest_profitability_audit(self) -> dict[str, Any] | None:
        path = latest_matching_file(self.profitability_dir, "*.json")
        if not path:
            return None
        payload = load_json(path)
        payload["_path"] = str(path)
        return payload

    def _latest_league_artifact(self, filename: str) -> dict[str, Any] | None:
        path = latest_matching_file(self.league_batches_dir, f"**/{filename}")
        if not path:
            return None
        payload = load_json(path, encoding="utf-8")
        payload["_path"] = str(path)
        return payload

    def _build_historical_context(self) -> dict[str, Any]:
        leaderboard = self._latest_leaderboard()
        audit = self._latest_profitability_audit()
        league = self._latest_league_artifact(LEAGUE_LEADERBOARD_FILENAME)
        learning = self._latest_league_artifact(LEAGUE_PROJECT_LEARNING_FILENAME)
        return {
            "latest_strategy_compare": {
                "path": leaderboard.get("_path") if leaderboard else None,
                "winner_family": (
                    leaderboard.get("selection", {}).get("winner_family") if leaderboard else None
                ),
                "winner_interval": (
                    leaderboard.get("selection", {}).get("winner_interval") if leaderboard else None
                ),
                "winner_verdict": (
                    leaderboard.get("selection", {}).get("winner_verdict") if leaderboard else None
                ),
            },
            "latest_profitability_audit": {
                "path": audit.get("_path") if audit else None,
                "verdict": audit.get("verdict") if audit else None,
                "profitability_score": audit.get("profitability_score") if audit else None,
            },
            "latest_league_batch": {
                "path": league.get("_path") if league else None,
                "batch_id": league.get("batch_id") if league else None,
                "snapshot_id": league.get("snapshot_id") if league else None,
                "gold_winner_family": (
                    league.get("selection", {}).get("gold_winner_family") if league else None
                ),
                "gold_winner_interval": (
                    league.get("selection", {}).get("gold_winner_interval") if league else None
                ),
                "baseline_family": (
                    league.get("selection", {}).get("baseline_family") if league else None
                ),
            },
            "latest_project_learning": {
                "path": learning.get("_path") if learning else None,
                "audit_verdict": learning.get("audit_verdict") if learning else None,
                "good_decisions": (
                    learning.get("patterns", {}).get("good_decisions", []) if learning else []
                ),
                "bad_decisions": (
                    learning.get("patterns", {}).get("bad_decisions", []) if learning else []
                ),
            },
        }

    def _build_league_context(self) -> dict[str, Any]:
        leaderboard = self._latest_league_artifact(LEAGUE_LEADERBOARD_FILENAME)
        audit = self._latest_league_artifact("league_audit.json")
        decision = self._latest_league_artifact(LEAGUE_DECISION_REVIEW_FILENAME)
        if not leaderboard:
            return {
                "status": "no_league_batch",
                "batch_id": None,
                "snapshot_id": None,
                "winner_family": None,
                "winner_interval": None,
                "baseline_family": None,
                "stable_winner": False,
                "audit_verdict": audit.get("verdict") if audit else None,
                "decision_summary": decision.get("decision_summary") if decision else None,
            }

        selection = leaderboard.get("selection", {})
        return {
            "status": "ok",
            "path": leaderboard.get("_path"),
            "batch_id": leaderboard.get("batch_id"),
            "snapshot_id": leaderboard.get("snapshot_id"),
            "winner_family": selection.get("gold_winner_family"),
            "winner_interval": selection.get("gold_winner_interval"),
            "baseline_family": selection.get("baseline_family"),
            "stable_winner": bool(selection.get("stable_winner", False)),
            "audit_verdict": audit.get("verdict") if audit else None,
            "decision_summary": decision.get("decision_summary") if decision else None,
        }

    def _build_project_learning_context(self) -> dict[str, Any]:
        learning = self._latest_league_artifact(LEAGUE_PROJECT_LEARNING_FILENAME)
        if not learning:
            return {
                "status": "no_project_learning",
                "batch_id": None,
                "snapshot_id": None,
                "history_alerts": [],
                "lesson_summary": [],
                "patterns": {
                    "good_decisions": [],
                    "bad_decisions": [],
                    "repeated_regressions": [],
                    "recovered_patterns": [],
                },
                "recommendations": [],
            }

        return {
            "status": "ok",
            "path": learning.get("_path"),
            "batch_id": learning.get("batch_id"),
            "snapshot_id": learning.get("snapshot_id"),
            "history_alerts": learning.get("history_alerts", []),
            "lesson_summary": learning.get("lesson_summary", []),
            "patterns": learning.get("patterns", {}),
            "recommendations": learning.get("recommendations", []),
        }

    def _build_diagnostics(
        self,
        session: DemoSessionArtifacts | None,
        broker: dict[str, Any],
    ) -> dict[str, Any]:
        if not session:
            return {
                "status": "no_session",
                "alerts": [{"level": "warning", "reason": "no_active_session"}],
                "rejections_by_reason": {},
                "signals_by_action": {},
                "drawdown": 0.0,
                "daily_loss": 0.0,
                "desync": None,
            }

        rejection_counts = Counter(
            event.get("reason", "unknown")
            for event in session.events
            if event.get("type") == "rejection"
        )
        signal_counts = Counter(
            event.get("action", "UNKNOWN")
            for event in session.events
            if event.get("type") == "signal"
        )
        latest_equity = session.equity[-1] if session.equity else {}
        peak_equity = max(
            (float(point.get("equity", 0.0)) for point in session.equity), default=0.0
        )
        current_equity = float(latest_equity.get("equity", 0.0))
        drawdown = (peak_equity - current_equity) / max(peak_equity, 1e-8) if peak_equity else 0.0

        today = datetime.now().date()
        today_points = [
            point
            for point in session.equity
            if (_parse_timestamp(point.get("timestamp")) or datetime.now()).date() == today
        ]
        daily_loss = 0.0
        if today_points:
            start_equity = float(today_points[0].get("equity", current_equity))
            daily_loss = max((start_equity - current_equity) / max(start_equity, 1e-8), 0.0)

        alerts: list[dict[str, Any]] = []
        live_settings = session.meta.get("live_settings", {})
        max_drawdown = float(live_settings.get("max_drawdown", 0.12))
        max_daily_loss = float(live_settings.get("max_daily_loss", 0.03))
        if drawdown >= max_drawdown * 0.8:
            alerts.append({"level": "warning", "reason": "drawdown_near_limit", "value": drawdown})
        if daily_loss >= max_daily_loss * 0.8:
            alerts.append(
                {"level": "warning", "reason": "daily_loss_near_limit", "value": daily_loss}
            )
        for reason in ("price_unavailable", "duplicate_signal", "observation_mismatch"):
            if rejection_counts.get(reason):
                alerts.append(
                    {"level": "warning", "reason": reason, "count": rejection_counts[reason]}
                )

        desync = None
        if broker.get("connected") and latest_equity:
            broker_equity = float(
                broker["account"].get("portfolio_value") or broker["account"].get("equity") or 0.0
            )
            logged_equity = float(latest_equity.get("equity", 0.0))
            delta = abs(broker_equity - logged_equity)
            desync = {
                "broker_equity": broker_equity,
                "logged_equity": logged_equity,
                "delta": delta,
            }
            if delta > max(250.0, broker_equity * 0.005):
                alerts.append(
                    {"level": "critical", "reason": "broker_journal_desync", "delta": delta}
                )

        return {
            "status": "ok",
            "alerts": alerts,
            "rejections_by_reason": dict(rejection_counts),
            "signals_by_action": dict(signal_counts),
            "drawdown": drawdown,
            "daily_loss": daily_loss,
            "desync": desync,
        }

    def _build_recommendations(
        self,
        session: DemoSessionArtifacts | None,
        diagnostics: dict[str, Any],
        historical_context: dict[str, Any],
        league_context: dict[str, Any],
        project_learning: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not session:
            return []

        recommendations: list[dict[str, Any]] = []
        rejection_counts = diagnostics.get("rejections_by_reason", {})
        latest_compare = historical_context["latest_strategy_compare"]
        latest_audit = historical_context["latest_profitability_audit"]
        live_interval = str(session.meta.get("interval", "1h"))

        if rejection_counts.get("observation_mismatch"):
            recommendations.append(
                {
                    "priority": 1,
                    "confidence": "high",
                    "title": "Fix live observation/model mismatch first",
                    "justification": "The demo is rejecting live snapshots due to observation size mismatch, so every strategy conclusion would be unreliable.",
                }
            )
        if rejection_counts.get("cost_too_high"):
            recommendations.append(
                {
                    "priority": 2,
                    "confidence": "high",
                    "title": "Reduce turnover or position sizing",
                    "justification": "Buy attempts are being blocked by estimated cost, which points to sizing or frequency that is too aggressive for the current execution model.",
                }
            )
        if (
            latest_compare.get("winner_interval")
            and latest_compare["winner_interval"] != live_interval
        ):
            recommendations.append(
                {
                    "priority": 3,
                    "confidence": "high",
                    "title": f"Evaluate the {latest_compare['winner_interval']} horizon in demo",
                    "justification": f"The latest strategy comparison favors {latest_compare['winner_family']} on {latest_compare['winner_interval']}, while the current demo session runs on {live_interval}.",
                }
            )
        if diagnostics.get("drawdown", 0.0) >= 0.08 or diagnostics.get("daily_loss", 0.0) >= 0.02:
            recommendations.append(
                {
                    "priority": 4,
                    "confidence": "medium",
                    "title": "Tighten live risk sizing",
                    "justification": "Current session drawdown pressure suggests reducing buy_pct or max_open_positions before looking for more raw edge.",
                }
            )
        if latest_audit.get("verdict") in {"improve_robustness", "improve_risk_profile"}:
            recommendations.append(
                {
                    "priority": 5,
                    "confidence": "medium",
                    "title": "Focus on robustness before chasing more return",
                    "justification": f"The latest profitability audit still says {latest_audit['verdict']}, so the next gains are more likely to come from robustness work than from extra model complexity.",
                }
            )
        if rejection_counts.get("low_confidence"):
            recommendations.append(
                {
                    "priority": 6,
                    "confidence": "medium",
                    "title": "Review live confidence threshold and feature stability",
                    "justification": "Repeated low-confidence blocks usually mean the live feature distribution is noisier than the training signal.",
                }
            )
        if league_context.get("winner_family"):
            recommendations.append(
                {
                    "priority": 7,
                    "confidence": "medium",
                    "title": f"Keep the live demo aligned with {league_context['winner_family']}",
                    "justification": "The latest league batch has already ranked a gold-stage winner; demo sessions should validate that candidate rather than drift to a different family.",
                }
            )
        for alert in project_learning.get("history_alerts", [])[:2]:
            recommendations.append(
                {
                    "priority": 8,
                    "confidence": "high",
                    "title": "Avoid repeating a known bad batch decision",
                    "justification": alert.get(
                        "message", "A previous batch already flagged this regression pattern."
                    ),
                }
            )

        recommendations.sort(key=lambda item: item["priority"])
        return recommendations[:5]

    def get_demo_payload(self) -> dict[str, Any]:
        session = self.sessions.latest()
        mode = session.meta.get("mode", "etoro") if session else "etoro"
        initial_balance = (
            float(session.report.get("initial_balance", 100_000.0)) if session else 100_000.0
        )
        broker = self.broker_cache.get_snapshot(mode=mode, initial_balance=initial_balance)
        diagnostics = self._build_diagnostics(session, broker)
        historical_context = self._build_historical_context()
        league_context = self._build_league_context()
        project_learning = self._build_project_learning_context()
        recommendations = self._build_recommendations(
            session,
            diagnostics,
            historical_context,
            league_context,
            project_learning,
        )
        return {
            "session": session,
            "broker": broker,
            "diagnostics": diagnostics,
            "historical_context": historical_context,
            "league_context": league_context,
            "project_learning": project_learning,
            "recommendations": recommendations,
        }
