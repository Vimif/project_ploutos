from __future__ import annotations

import json
from pathlib import Path

from dashboard.demo_monitor import DemoMonitorService


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _build_session(root: Path) -> None:
    session_dir = root / "20260404_120000"
    _write_json(
        session_dir / "session_meta.json",
        {
            "session_id": session_dir.name,
            "mode": "etoro",
            "broker": "etoro",
            "interval": "1h",
            "tickers": ["AAPL", "SPY"],
            "live_settings": {"max_drawdown": 0.12, "max_daily_loss": 0.03},
        },
    )
    _write_jsonl(
        session_dir / "events.jsonl",
        [
            {
                "timestamp": "2026-04-04T12:00:00",
                "type": "signal",
                "ticker": "AAPL",
                "action": "BUY",
                "reason": "model_buy",
            },
            {
                "timestamp": "2026-04-04T12:00:01",
                "type": "rejection",
                "ticker": "AAPL",
                "action": "BUY",
                "reason": "cost_too_high",
            },
            {
                "timestamp": "2026-04-04T12:00:02",
                "type": "rejection",
                "ticker": "AAPL",
                "action": "BUY",
                "reason": "low_confidence",
            },
        ],
    )
    _write_jsonl(
        session_dir / "equity.jsonl",
        [
            {
                "timestamp": "2026-04-04T09:00:00",
                "equity": 100000.0,
                "balance": 100000.0,
                "n_positions": 0,
                "drawdown": 0.0,
                "exposure": 0.0,
            },
            {
                "timestamp": "2026-04-04T12:00:03",
                "equity": 98000.0,
                "balance": 98000.0,
                "n_positions": 0,
                "drawdown": 0.02,
                "exposure": 0.0,
            },
        ],
    )
    _write_json(
        session_dir / "report.json",
        {
            "session_id": session_dir.name,
            "initial_balance": 100000.0,
            "summary": {
                "n_trades": 0,
                "n_rejections": 2,
                "total_return": -0.02,
                "final_equity": 98000.0,
            },
        },
    )


def test_demo_monitor_recommendations_blend_live_and_historical_context(monkeypatch, tmp_path):
    paper_dir = tmp_path / "paper"
    compare_dir = tmp_path / "compare"
    profitability_dir = tmp_path / "profitability"
    league_dir = tmp_path / "league"
    _build_session(paper_dir)
    _write_json(
        compare_dir / "fast_real_run" / "strategy_leaderboard.json",
        {
            "selection": {
                "winner_family": "rule_momentum_regime",
                "winner_interval": "4h",
                "winner_verdict": "improve_robustness",
            }
        },
    )
    _write_json(
        profitability_dir / "audit_latest.json",
        {"verdict": "improve_robustness", "profitability_score": 68.0},
    )
    _write_json(
        league_dir / "20260404_140000" / "league_leaderboard.json",
        {
            "batch_id": "20260404_140000",
            "snapshot_id": "snapshot_test",
            "selection": {
                "gold_winner_family": "supervised_ranker",
                "gold_winner_interval": "4h",
                "baseline_family": "rule_momentum_regime",
            },
        },
    )
    _write_json(
        league_dir / "20260404_140000" / "project_learning.json",
        {
            "batch_id": "20260404_140000",
            "snapshot_id": "snapshot_test",
            "audit_verdict": "needs_iteration",
            "history_alerts": [
                {
                    "level": "warning",
                    "decision_key": "ppo_ensemble@4h",
                    "message": "ppo_ensemble@4h is repeating a previously bad decision pattern.",
                }
            ],
            "patterns": {
                "good_decisions": ["supervised_ranker@4h"],
                "bad_decisions": ["ppo_ensemble@4h"],
            },
            "lesson_summary": ["Avoid candidate patterns: ppo_ensemble@4h."],
            "recommendations": ["Improve Monte Carlo resilience before the next demo batch."],
        },
    )

    service = DemoMonitorService(
        paper_trading_dir=paper_dir,
        strategy_compare_dir=compare_dir,
        profitability_dir=profitability_dir,
        league_batches_dir=league_dir,
    )
    monkeypatch.setattr(
        service.broker_cache,
        "get_snapshot",
        lambda **kwargs: {
            "connected": False,
            "account": {},
            "positions": [],
            "open_orders": [],
            "error": "offline",
            "mode": "etoro",
        },
    )

    payload = service.get_demo_payload()
    titles = [item["title"] for item in payload["recommendations"]]

    assert payload["diagnostics"]["rejections_by_reason"]["cost_too_high"] == 1
    assert any("4h horizon" in title for title in titles)
    assert any("Reduce turnover" in title for title in titles)
    assert payload["league_context"]["winner_family"] == "supervised_ranker"
    assert payload["project_learning"]["history_alerts"]
