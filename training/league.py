"""Versioned league batches for profit/risk strategy evaluation."""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.artifacts import (
    LEAGUE_AUDIT_FILENAME,
    LEAGUE_DECISION_REVIEW_FILENAME,
    LEAGUE_DEMO_FOLLOWUP_FILENAME,
    LEAGUE_LEADERBOARD_FILENAME,
    LEAGUE_PROJECT_LEARNING_FILENAME,
    latest_demo_session,
    load_json,
    save_json,
)
from scripts.profitability_audit import build_profitability_report
from training import strategy_compare
from training.strategy_compare import (
    _beats_baseline,
    _select_phase2_candidates,
    build_protocol_snapshot,
    evaluate_candidate_family,
    generate_walk_forward_splits,
    load_compare_config,
    load_market_bundle,
    promotion_thresholds_from_config,
)

DEFAULT_LEAGUE_OUTPUT_DIR = Path("logs") / "league_batches"
DEFAULT_LEAGUE_FAMILIES = ["supervised_ranker", "ppo_ensemble", "rule_momentum_regime"]


def _dedupe_families(families: list[str]) -> list[str]:
    unique: list[str] = []
    for family in families:
        normalized = str(family).strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def resolve_league_settings(
    config: dict,
    *,
    batch_id: str | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Resolve league defaults from config and runtime overrides."""

    league_cfg = dict(config.get("league", {}))
    tickers = list(config.get("data", {}).get("tickers", []))
    interval = str(config.get("data", {}).get("interval", "4h"))
    gold_holdout_months = int(league_cfg.get("gold_holdout_months", 6))
    resolved_snapshot_id = snapshot_id or league_cfg.get(
        "snapshot_id",
        f"{interval}_{len(tickers)}assets_gold{gold_holdout_months}m",
    )
    resolved_batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "snapshot_id": str(resolved_snapshot_id),
        "batch_id": str(resolved_batch_id),
        "candidate_families": _dedupe_families(
            list(league_cfg.get("candidate_families", DEFAULT_LEAGUE_FAMILIES))
        ),
        "baseline_family": str(league_cfg.get("baseline_family", "rule_momentum_regime")),
        "gold_holdout_months": gold_holdout_months,
        "batch_output_root": Path(
            league_cfg.get("batch_output_root", str(DEFAULT_LEAGUE_OUTPUT_DIR))
        ),
        "cadence": str(league_cfg.get("cadence", "batch_complete_only")),
        "learning_granularity": str(league_cfg.get("learning_granularity", "batch_decision")),
        "learning_action_mode": str(league_cfg.get("learning_action_mode", "advisory")),
    }


def _reference_bounds(data: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    ref_ticker = next(iter(data))
    ref_df = data[ref_ticker]
    return ref_df.index[0], ref_df.index[-1]


def _trim_data_to_end(
    data: dict[str, pd.DataFrame],
    *,
    end_exclusive: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    trimmed: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        current = df.loc[df.index < end_exclusive].copy()
        if not current.empty:
            trimmed[ticker] = current
    return trimmed


def _build_fixed_holdout_splits(
    data: dict[str, pd.DataFrame],
    *,
    holdout_start: pd.Timestamp,
) -> list[dict[str, Any]]:
    train_slice: dict[str, pd.DataFrame] = {}
    test_slice: dict[str, pd.DataFrame] = {}
    for ticker, df in data.items():
        current_train = df.loc[df.index < holdout_start].copy()
        current_test = df.loc[df.index >= holdout_start].copy()
        if len(current_train) < 100 or len(current_test) < 50:
            return []
        train_slice[ticker] = current_train
        test_slice[ticker] = current_test

    ref_ticker = next(iter(data))
    ref_df = data[ref_ticker]
    return [
        {
            "train": train_slice,
            "test": test_slice,
            "train_start": str(ref_df.index[0].date()),
            "train_end": str((holdout_start - pd.Timedelta(seconds=1)).date()),
            "test_start": str(holdout_start.date()),
            "test_end": str(ref_df.index[-1].date()),
        }
    ]


def _snapshot_manifest(
    *,
    snapshot_id: str,
    current_interval: str,
    phase2_interval: str,
    current_data: dict[str, pd.DataFrame],
    holdout_start: pd.Timestamp,
    gold_holdout_months: int,
) -> dict[str, Any]:
    start, end = _reference_bounds(current_data)
    challenge_end = holdout_start - pd.Timedelta(seconds=1)
    return {
        "snapshot_id": snapshot_id,
        "interval": current_interval,
        "phase2_interval": phase2_interval,
        "gold_holdout_months": int(gold_holdout_months),
        "dataset_start": str(start.date()),
        "dataset_end": str(end.date()),
        "challenge": {
            "start": str(start.date()),
            "end": str(challenge_end.date()),
        },
        "gold": {
            "start": str(holdout_start.date()),
            "end": str(end.date()),
        },
    }


def _candidate_lesson_summary(candidate: dict[str, Any]) -> str:
    verdict = candidate.get("verdict", "needs_iteration")
    if verdict == "promotable_demo":
        return "This candidate currently clears the stage promotion gates."
    if verdict == "improve_robustness":
        return "This candidate shows some edge, but Monte Carlo or crash resilience still lags."
    if verdict == "improve_risk_profile":
        return "This candidate needs tighter risk before its return can be trusted."
    if verdict == "improve_edge":
        return "This candidate is not generating enough risk-adjusted edge yet."
    if verdict == "suspect_artifact":
        return "This candidate is flagged as suspect and should not drive live decisions."
    return "This candidate remains a valid research challenger but is not yet ready for promotion."


def _annotate_candidates(
    candidates: list[dict[str, Any]],
    *,
    batch_id: str,
    snapshot_id: str,
    evaluation_stage: str,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for candidate in candidates:
        enriched = dict(candidate)
        enriched["batch_id"] = batch_id
        enriched["snapshot_id"] = snapshot_id
        enriched["candidate_family"] = candidate["family"]
        enriched["evaluation_stage"] = evaluation_stage
        enriched["decision_summary"] = (
            f"{candidate['family']} on {candidate['interval']} -> "
            f"{candidate['verdict']} ({candidate['selection_score']}/100)"
        )
        enriched["lesson_summary"] = _candidate_lesson_summary(candidate)
        annotated.append(enriched)
    return annotated


def _find_candidate(
    candidates: list[dict[str, Any]],
    family: str,
    interval: str | None = None,
) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate["family"] != family:
            continue
        if interval is None or candidate["interval"] == interval:
            return candidate
    return None


def _find_stage_winner(stage_payload: dict[str, Any]) -> dict[str, Any] | None:
    winner_family = stage_payload.get("selection", {}).get("winner_family")
    winner_interval = stage_payload.get("selection", {}).get("winner_interval")
    candidates = list(stage_payload.get("phase_2", [])) or list(stage_payload.get("phase_1", []))
    return _find_candidate(candidates, winner_family, winner_interval)


def _select_phase2_with_baseline(
    phase1_candidates: list[dict[str, Any]],
    *,
    top_k: int,
    baseline_family: str,
) -> list[str]:
    selected = _select_phase2_candidates(phase1_candidates, top_k)
    if baseline_family not in selected:
        selected.append(baseline_family)
    return _dedupe_families(selected)


def _build_stage_leaderboard(
    *,
    config: dict,
    stage_name: str,
    batch_id: str,
    snapshot_id: str,
    families: list[str],
    baseline_family: str,
    current_interval: str,
    current_macro: pd.DataFrame | None,
    current_splits: list[dict[str, Any]],
    phase2_interval: str,
    phase2_macro: pd.DataFrame | None,
    phase2_splits: list[dict[str, Any]],
    phase2_top_k: int,
    monte_carlo_sims: int,
    monte_carlo_noise_std: float,
    extreme_return_threshold: float,
    seed_offsets: list[int],
    snapshot_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one league stage across challenge or gold windows."""

    protocol = build_protocol_snapshot(
        config,
        interval=current_interval,
        splits=current_splits,
        families=families,
        strategy_settings={
            "phase2_interval": phase2_interval,
            "phase2_top_k": phase2_top_k,
            "monte_carlo_sims": monte_carlo_sims,
            "monte_carlo_noise_std": monte_carlo_noise_std,
            "extreme_return_threshold": extreme_return_threshold,
            "seed_offsets": seed_offsets,
        },
    )
    protocol["snapshot_id"] = snapshot_id
    protocol["batch_id"] = batch_id
    protocol["evaluation_stage"] = stage_name
    protocol["baseline_family"] = baseline_family
    protocol["snapshot_manifest"] = snapshot_manifest

    phase1 = [
        evaluate_candidate_family(
            family,
            config=config,
            splits=current_splits,
            macro_data=current_macro,
            interval=current_interval,
            monte_carlo_sims=monte_carlo_sims,
            monte_carlo_noise_std=monte_carlo_noise_std,
            extreme_return_threshold=extreme_return_threshold,
            seed_offsets=seed_offsets,
        )
        for family in families
    ]
    phase1 = sorted(
        _annotate_candidates(
            phase1,
            batch_id=batch_id,
            snapshot_id=snapshot_id,
            evaluation_stage=stage_name,
        ),
        key=lambda item: item["selection_score"],
        reverse=True,
    )

    phase2: list[dict[str, Any]] = []
    if phase2_interval != current_interval and phase2_splits:
        phase2_config = copy.deepcopy(config)
        phase2_config.setdefault("data", {})["interval"] = phase2_interval
        phase2_families = _select_phase2_with_baseline(
            phase1,
            top_k=phase2_top_k,
            baseline_family=baseline_family,
        )
        phase2 = [
            evaluate_candidate_family(
                family,
                config=phase2_config,
                splits=phase2_splits,
                macro_data=phase2_macro,
                interval=phase2_interval,
                monte_carlo_sims=monte_carlo_sims,
                monte_carlo_noise_std=monte_carlo_noise_std,
                extreme_return_threshold=min(extreme_return_threshold, 3.0),
                seed_offsets=seed_offsets,
            )
            for family in phase2_families
        ]
        phase2 = sorted(
            _annotate_candidates(
                phase2,
                batch_id=batch_id,
                snapshot_id=snapshot_id,
                evaluation_stage=stage_name,
            ),
            key=lambda item: item["selection_score"],
            reverse=True,
        )

    winning_pool = phase2 or phase1
    winner = winning_pool[0] if winning_pool else None
    baseline = None
    if winner is not None:
        baseline = _find_candidate(winning_pool, baseline_family, winner["interval"])
    if baseline is None:
        baseline = _find_candidate(phase1, baseline_family)

    winner_beats_baseline = bool(winner and baseline and _beats_baseline(winner, baseline))
    thresholds = promotion_thresholds_from_config(config)
    return {
        "created_at": datetime.now().isoformat(),
        "batch_id": batch_id,
        "snapshot_id": snapshot_id,
        "evaluation_stage": stage_name,
        "protocol": protocol,
        "phase_1": phase1,
        "phase_2": phase2,
        "selection": {
            "winner_family": winner["family"] if winner else None,
            "winner_interval": winner["interval"] if winner else None,
            "winner_selection_score": winner["selection_score"] if winner else None,
            "winner_verdict": winner["verdict"] if winner else None,
            "winner_beats_baseline": winner_beats_baseline,
            "baseline_family": baseline["family"] if baseline else baseline_family,
            "baseline_interval": baseline["interval"] if baseline else None,
            "baseline_selection_score": baseline["selection_score"] if baseline else None,
            "promotion_thresholds": thresholds,
        },
    }


def _load_latest_demo_session(paper_trading_dir: Path) -> dict[str, Any] | None:
    session = latest_demo_session(paper_trading_dir)
    if session is None:
        return None
    return {
        "session_id": session.session_id,
        "session_dir": str(session.session_dir),
        "report": session.report,
        "meta": session.meta,
    }


def _candidate_delta(candidate: dict, baseline: dict) -> dict[str, float]:
    return {
        "selection_score": float(candidate["selection_score"] - baseline["selection_score"]),
        "cumulative_return": float(
            candidate["walk_forward"]["cumulative_return"]
            - baseline["walk_forward"]["cumulative_return"]
        ),
        "avg_sharpe": float(
            candidate["walk_forward"]["avg_sharpe"] - baseline["walk_forward"]["avg_sharpe"]
        ),
        "avg_max_drawdown": float(
            candidate["walk_forward"]["avg_max_drawdown"]
            - baseline["walk_forward"]["avg_max_drawdown"]
        ),
        "loss_rate": float(
            candidate["robustness"]["monte_carlo"]["loss_rate"]
            - baseline["robustness"]["monte_carlo"]["loss_rate"]
        ),
    }


def build_demo_followup(
    *,
    batch_id: str,
    snapshot_id: str,
    expected_candidate: dict[str, Any] | None,
    paper_trading_dir: str | Path = "logs/paper_trading",
) -> dict[str, Any]:
    """Summarize whether the latest demo session validates the current league winner."""

    session = _load_latest_demo_session(Path(paper_trading_dir))
    expected_family = expected_candidate["family"] if expected_candidate else None
    if not session:
        return {
            "created_at": datetime.now().isoformat(),
            "batch_id": batch_id,
            "snapshot_id": snapshot_id,
            "candidate_family": expected_family,
            "evaluation_stage": "demo",
            "status": "no_demo_session",
            "decision_summary": "No demo session is available for this batch yet.",
            "lesson_summary": "Live validation is still pending.",
            "latest_session": None,
        }

    session_family = session["report"].get("strategy_family") or session["meta"].get(
        "strategy_family"
    )
    aligned = bool(
        expected_family
        and session_family == expected_family
        and str(session["report"].get("mode") or session["meta"].get("mode", "")).lower() == "etoro"
    )
    status = "aligned_demo_candidate" if aligned else "demo_needs_refresh"
    summary = session["report"].get("summary", {})
    return {
        "created_at": datetime.now().isoformat(),
        "batch_id": batch_id,
        "snapshot_id": snapshot_id,
        "candidate_family": expected_family,
        "evaluation_stage": "demo",
        "status": status,
        "decision_summary": (
            f"Latest demo session uses {session_family or 'unknown'} and "
            f"{'matches' if aligned else 'does not match'} the current gold winner."
        ),
        "lesson_summary": (
            "The latest demo session is aligned with the current winner."
            if aligned
            else "The demo run should be refreshed with the current gold winner before promotion."
        ),
        "latest_session": {
            "session_id": session["session_id"],
            "session_dir": session["session_dir"],
            "mode": session["report"].get("mode") or session["meta"].get("mode"),
            "strategy_family": session_family,
            "summary": summary,
        },
    }


def build_league_audit(
    *,
    config: dict,
    batch_id: str,
    snapshot_id: str,
    challenge_stage: dict[str, Any],
    gold_stage: dict[str, Any],
    demo_followup: dict[str, Any],
) -> dict[str, Any]:
    """Create an audit centered on the gold-stage winner and its baseline delta."""

    gold_winner = _find_stage_winner(gold_stage)
    challenge_winner = _find_stage_winner(challenge_stage)
    if gold_winner is None:
        raise RuntimeError("Gold stage produced no winner")

    baseline_family = gold_stage["selection"]["baseline_family"]
    baseline = _find_candidate(
        gold_stage.get("phase_2", []) or gold_stage.get("phase_1", []),
        baseline_family,
        gold_winner["interval"],
    ) or _find_candidate(gold_stage.get("phase_1", []), baseline_family)
    if baseline is None:
        raise RuntimeError("Gold stage baseline is unavailable")

    audit = build_profitability_report(
        walk_forward_results=gold_winner["walk_forward"],
        robustness_reports=[gold_winner["robustness"]],
        config=config,
    )
    beats_baseline = _beats_baseline(gold_winner, baseline)
    audit.update(
        {
            "batch_id": batch_id,
            "snapshot_id": snapshot_id,
            "candidate_family": gold_winner["family"],
            "candidate_interval": gold_winner["interval"],
            "evaluation_stage": "gold",
            "challenge_winner_family": challenge_winner["family"] if challenge_winner else None,
            "gold_winner_family": gold_winner["family"],
            "gold_winner_beats_baseline": beats_baseline,
            "baseline_family": baseline["family"],
            "baseline_interval": baseline["interval"],
            "baseline_comparison": {
                "delta": _candidate_delta(gold_winner, baseline),
                "baseline_verdict": baseline["verdict"],
                "baseline_selection_score": baseline["selection_score"],
            },
            "demo_followup_status": demo_followup["status"],
            "decision_summary": (
                f"Gold winner {gold_winner['family']} on {gold_winner['interval']} "
                f"{'beats' if beats_baseline else 'does not beat'} the baseline {baseline['family']}."
            ),
            "lesson_summary": (
                "The gold winner is the best current candidate for demo follow-up."
                if beats_baseline
                else "The baseline still offers the most trustworthy risk-adjusted profile."
            ),
        }
    )
    return audit


def _extract_learning_focus(config: dict) -> dict[str, Any]:
    live_cfg = config.get("live", {})
    return {
        "data_interval": config.get("data", {}).get("interval"),
        "buy_pct": live_cfg.get("buy_pct"),
        "max_position_pct": live_cfg.get("max_position_pct"),
        "max_open_positions": live_cfg.get("max_open_positions"),
        "stop_loss_pct": live_cfg.get("stop_loss_pct"),
        "take_profit_pct": live_cfg.get("take_profit_pct"),
        "max_cost_pct": live_cfg.get("max_cost_pct"),
    }


def _probable_causes(candidate: dict, baseline: dict) -> list[str]:
    causes: list[str] = []
    if candidate["walk_forward"]["avg_sharpe"] < baseline["walk_forward"]["avg_sharpe"]:
        causes.append("sharpe_below_baseline")
    if candidate["walk_forward"]["avg_max_drawdown"] > baseline["walk_forward"]["avg_max_drawdown"]:
        causes.append("drawdown_above_baseline")
    if (
        candidate["robustness"]["monte_carlo"]["loss_rate"]
        > baseline["robustness"]["monte_carlo"]["loss_rate"]
    ):
        causes.append("loss_rate_above_baseline")
    if candidate["verdict"] == "improve_edge":
        causes.append("insufficient_edge")
    if candidate["verdict"] == "improve_robustness":
        causes.append("robustness_gap")
    if candidate["verdict"] == "improve_risk_profile":
        causes.append("risk_profile_gap")
    if candidate["verdict"] == "suspect_artifact":
        causes.append("suspect_artifact")
    return causes or ["mixed_signal"]


def _record_verdict(candidate: dict, baseline: dict) -> str:
    if candidate["evidence_status"] == "suspect":
        return "bad_decision"
    if candidate["verdict"] in {"improve_edge", "improve_risk_profile", "suspect_artifact"}:
        return "bad_decision"

    cumulative_return = float(candidate["walk_forward"]["cumulative_return"])
    avg_sharpe = float(candidate["walk_forward"]["avg_sharpe"])
    if cumulative_return <= 0 or avg_sharpe <= 0:
        return "bad_decision"

    if not _beats_baseline(candidate, baseline):
        return "inconclusive"

    if candidate["verdict"] in {"promotable_demo", "needs_iteration"}:
        return "good_decision"
    if candidate["verdict"] == "improve_robustness":
        return "inconclusive"
    return "inconclusive"


def _next_batch_recommendations(candidate: dict, baseline: dict) -> list[str]:
    recommendations: list[str] = []
    if candidate["walk_forward"]["avg_max_drawdown"] > baseline["walk_forward"]["avg_max_drawdown"]:
        recommendations.append("Tighten sizing or stop loss before promoting this candidate.")
    if (
        candidate["robustness"]["monte_carlo"]["loss_rate"]
        > baseline["robustness"]["monte_carlo"]["loss_rate"]
    ):
        recommendations.append("Improve Monte Carlo resilience before the next demo batch.")
    if candidate["walk_forward"]["avg_sharpe"] < baseline["walk_forward"]["avg_sharpe"]:
        recommendations.append("Improve edge quality before increasing complexity.")
    if not recommendations:
        recommendations.append("Retest this candidate on the next batch with the same protocol.")
    return recommendations


def _load_previous_project_learning(
    batch_root: Path,
    *,
    current_batch_id: str,
) -> dict[str, Any] | None:
    candidates = sorted(
        batch_root.glob(f"**/{LEAGUE_PROJECT_LEARNING_FILENAME}"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if candidate.parent.name == current_batch_id:
            continue
        payload = load_json(candidate, encoding="utf-8")
        payload["_path"] = str(candidate)
        return payload
    return None


def build_project_learning(
    *,
    config: dict,
    batch_id: str,
    snapshot_id: str,
    batch_root: Path,
    gold_stage: dict[str, Any],
    league_audit: dict[str, Any],
) -> dict[str, Any]:
    """Build project memory with batch-level lessons and repeated-error alerts."""

    previous = _load_previous_project_learning(batch_root, current_batch_id=batch_id)
    candidate_pool = list(gold_stage.get("phase_2", [])) or list(gold_stage.get("phase_1", []))
    baseline_family = gold_stage["selection"]["baseline_family"]
    baseline_candidates = {
        candidate["interval"]: candidate
        for candidate in candidate_pool
        if candidate["family"] == baseline_family
    }
    if not baseline_candidates:
        baseline_phase1 = _find_candidate(gold_stage.get("phase_1", []), baseline_family)
        if baseline_phase1:
            baseline_candidates[baseline_phase1["interval"]] = baseline_phase1

    decision_records: list[dict[str, Any]] = []
    for candidate in candidate_pool:
        if candidate["family"] == baseline_family:
            continue
        baseline = baseline_candidates.get(candidate["interval"]) or next(
            iter(baseline_candidates.values()), None
        )
        if baseline is None:
            continue
        verdict = _record_verdict(candidate, baseline)
        decision_records.append(
            {
                "decision_key": f"{candidate['family']}@{candidate['interval']}",
                "candidate_family": candidate["family"],
                "candidate_interval": candidate["interval"],
                "evaluation_stage": "gold",
                "tested_decision": f"Compare {candidate['family']} against {baseline_family} on {candidate['interval']}",
                "changes_applied": _extract_learning_focus(config)
                | {"candidate_family": candidate["family"]},
                "metrics_before": {
                    "baseline_family": baseline["family"],
                    "selection_score": baseline["selection_score"],
                    "cumulative_return": baseline["walk_forward"]["cumulative_return"],
                    "avg_sharpe": baseline["walk_forward"]["avg_sharpe"],
                    "avg_max_drawdown": baseline["walk_forward"]["avg_max_drawdown"],
                    "loss_rate": baseline["robustness"]["monte_carlo"]["loss_rate"],
                },
                "metrics_after": {
                    "selection_score": candidate["selection_score"],
                    "cumulative_return": candidate["walk_forward"]["cumulative_return"],
                    "avg_sharpe": candidate["walk_forward"]["avg_sharpe"],
                    "avg_max_drawdown": candidate["walk_forward"]["avg_max_drawdown"],
                    "loss_rate": candidate["robustness"]["monte_carlo"]["loss_rate"],
                },
                "verdict": verdict,
                "probable_causes": _probable_causes(candidate, baseline),
                "recommendations": _next_batch_recommendations(candidate, baseline),
                "decision_summary": candidate["decision_summary"],
                "lesson_summary": candidate["lesson_summary"],
            }
        )

    previous_records = {
        record.get("decision_key"): record
        for record in (previous or {}).get("decision_records", [])
        if record.get("decision_key")
    }
    history_alerts: list[dict[str, Any]] = []
    regression_patterns: list[str] = []
    recovered_patterns: list[str] = []
    for record in decision_records:
        previous_record = previous_records.get(record["decision_key"])
        if not previous_record:
            continue
        if (
            previous_record.get("verdict") == "bad_decision"
            and record["verdict"] != "good_decision"
        ):
            history_alerts.append(
                {
                    "level": "warning",
                    "decision_key": record["decision_key"],
                    "message": f"{record['decision_key']} is repeating a previously bad decision pattern.",
                }
            )
            regression_patterns.append(record["decision_key"])
        elif (
            previous_record.get("verdict") == "good_decision"
            and record["verdict"] == "bad_decision"
        ):
            history_alerts.append(
                {
                    "level": "critical",
                    "decision_key": record["decision_key"],
                    "message": f"{record['decision_key']} regressed from a previously good pattern.",
                }
            )
            regression_patterns.append(record["decision_key"])
        elif (
            previous_record.get("verdict") == "bad_decision"
            and record["verdict"] == "good_decision"
        ):
            recovered_patterns.append(record["decision_key"])

    good_patterns = [
        record["decision_key"]
        for record in decision_records
        if record["verdict"] == "good_decision"
    ]
    bad_patterns = [
        record["decision_key"] for record in decision_records if record["verdict"] == "bad_decision"
    ]
    lesson_summary: list[str] = []
    if good_patterns:
        lesson_summary.append(f"Repeat candidate patterns: {', '.join(sorted(good_patterns))}.")
    if bad_patterns:
        lesson_summary.append(f"Avoid candidate patterns: {', '.join(sorted(bad_patterns))}.")
    if not lesson_summary:
        lesson_summary.append("Current batch remains inconclusive; keep the baseline as reference.")

    recommendations: list[str] = []
    for record in decision_records:
        recommendations.extend(record["recommendations"])
    recommendations = list(dict.fromkeys(recommendations))
    return {
        "created_at": datetime.now().isoformat(),
        "batch_id": batch_id,
        "snapshot_id": snapshot_id,
        "candidate_family": league_audit.get("candidate_family"),
        "evaluation_stage": "project_learning",
        "learning_granularity": "batch_decision",
        "learning_action_mode": "advisory",
        "history_source": previous.get("_path") if previous else None,
        "history_alerts": history_alerts,
        "decision_records": decision_records,
        "patterns": {
            "good_decisions": sorted(good_patterns),
            "bad_decisions": sorted(bad_patterns),
            "repeated_regressions": sorted(set(regression_patterns)),
            "recovered_patterns": sorted(set(recovered_patterns)),
        },
        "lesson_summary": lesson_summary,
        "recommendations": recommendations[:8],
        "audit_verdict": league_audit["verdict"],
        "decision_summary": f"Project memory updated for batch {batch_id}.",
    }


def build_decision_review(
    *,
    batch_id: str,
    snapshot_id: str,
    challenge_stage: dict[str, Any],
    gold_stage: dict[str, Any],
    league_audit: dict[str, Any],
    demo_followup: dict[str, Any],
    project_learning: dict[str, Any],
) -> dict[str, Any]:
    """Create a compact batch decision summary."""

    challenge_winner = _find_stage_winner(challenge_stage)
    gold_winner = _find_stage_winner(gold_stage)
    recommended_family = (
        gold_winner["family"]
        if gold_winner and league_audit.get("gold_winner_beats_baseline")
        else gold_stage["selection"]["baseline_family"]
    )
    return {
        "created_at": datetime.now().isoformat(),
        "batch_id": batch_id,
        "snapshot_id": snapshot_id,
        "challenge_winner_family": challenge_winner["family"] if challenge_winner else None,
        "gold_winner_family": gold_winner["family"] if gold_winner else None,
        "candidate_family": recommended_family,
        "evaluation_stage": "batch_review",
        "decision_summary": (
            f"Recommended candidate for the next demo step: {recommended_family}. "
            f"Gold verdict is {league_audit['verdict']} and demo status is {demo_followup['status']}."
        ),
        "lesson_summary": " ".join(project_learning.get("lesson_summary", [])),
        "approved_change_window": "after_batch_complete_only",
        "history_alerts": project_learning.get("history_alerts", []),
        "recommendations": project_learning.get("recommendations", [])[:5],
    }


def run_league_batch(
    *,
    config_path: str = "config/config.yaml",
    output_dir: str | None = None,
    batch_id: str | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Run one complete league batch and persist all canonical outputs."""

    config = load_compare_config(config_path)
    settings = resolve_league_settings(config, batch_id=batch_id, snapshot_id=snapshot_id)
    batch_output_root = settings["batch_output_root"]
    batch_dir = Path(output_dir) if output_dir else batch_output_root / settings["batch_id"]
    current_interval = str(config.get("data", {}).get("interval", "4h"))
    phase2_interval = str(config.get("strategy", {}).get("phase2_interval", current_interval))
    families = settings["candidate_families"]

    current_data, current_macro = load_market_bundle(config, interval=current_interval)
    _, current_end = _reference_bounds(current_data)
    holdout_start = current_end - pd.DateOffset(months=settings["gold_holdout_months"])
    challenge_current_data = _trim_data_to_end(current_data, end_exclusive=holdout_start)

    wf_cfg = config.get("walk_forward", {})
    challenge_current_splits = generate_walk_forward_splits(
        challenge_current_data,
        train_years=int(wf_cfg.get("train_years", 3)),
        test_months=int(wf_cfg.get("test_months", 6)),
        step_months=int(wf_cfg.get("step_months", 6)),
    )
    if not challenge_current_splits:
        raise RuntimeError("No challenge-stage splits available for the configured league batch")

    gold_current_splits = _build_fixed_holdout_splits(current_data, holdout_start=holdout_start)
    if not gold_current_splits:
        raise RuntimeError("No gold-stage holdout split available for the configured league batch")

    if phase2_interval == current_interval:
        phase2_macro = current_macro
        challenge_phase2_splits: list[dict[str, Any]] = []
        gold_phase2_splits: list[dict[str, Any]] = []
    else:
        phase2_data, phase2_macro = load_market_bundle(config, interval=phase2_interval)
        challenge_phase2_data = _trim_data_to_end(phase2_data, end_exclusive=holdout_start)
        challenge_phase2_splits = generate_walk_forward_splits(
            challenge_phase2_data,
            train_years=int(wf_cfg.get("train_years", 3)),
            test_months=int(wf_cfg.get("test_months", 6)),
            step_months=int(wf_cfg.get("step_months", 6)),
        )
        gold_phase2_splits = _build_fixed_holdout_splits(phase2_data, holdout_start=holdout_start)

    snapshot_manifest = _snapshot_manifest(
        snapshot_id=settings["snapshot_id"],
        current_interval=current_interval,
        phase2_interval=phase2_interval,
        current_data=current_data,
        holdout_start=holdout_start,
        gold_holdout_months=settings["gold_holdout_months"],
    )

    strategy_settings = strategy_compare.resolve_strategy_settings(config)
    challenge_stage = _build_stage_leaderboard(
        config=config,
        stage_name="challenge",
        batch_id=settings["batch_id"],
        snapshot_id=settings["snapshot_id"],
        families=families,
        baseline_family=settings["baseline_family"],
        current_interval=current_interval,
        current_macro=current_macro,
        current_splits=challenge_current_splits,
        phase2_interval=phase2_interval,
        phase2_macro=phase2_macro,
        phase2_splits=challenge_phase2_splits,
        phase2_top_k=int(strategy_settings["phase2_top_k"]),
        monte_carlo_sims=int(strategy_settings["monte_carlo_sims"]),
        monte_carlo_noise_std=float(strategy_settings["monte_carlo_noise_std"]),
        extreme_return_threshold=float(strategy_settings["extreme_return_threshold"]),
        seed_offsets=list(strategy_settings["seed_offsets"]),
        snapshot_manifest=snapshot_manifest,
    )
    gold_stage = _build_stage_leaderboard(
        config=config,
        stage_name="gold",
        batch_id=settings["batch_id"],
        snapshot_id=settings["snapshot_id"],
        families=families,
        baseline_family=settings["baseline_family"],
        current_interval=current_interval,
        current_macro=current_macro,
        current_splits=gold_current_splits,
        phase2_interval=phase2_interval,
        phase2_macro=phase2_macro,
        phase2_splits=gold_phase2_splits,
        phase2_top_k=int(strategy_settings["phase2_top_k"]),
        monte_carlo_sims=int(strategy_settings["monte_carlo_sims"]),
        monte_carlo_noise_std=float(strategy_settings["monte_carlo_noise_std"]),
        extreme_return_threshold=float(strategy_settings["extreme_return_threshold"]),
        seed_offsets=list(strategy_settings["seed_offsets"]),
        snapshot_manifest=snapshot_manifest,
    )

    challenge_winner = _find_stage_winner(challenge_stage)
    gold_winner = _find_stage_winner(gold_stage)
    league_leaderboard = {
        "created_at": datetime.now().isoformat(),
        "batch_id": settings["batch_id"],
        "snapshot_id": settings["snapshot_id"],
        "candidate_family": gold_winner["family"] if gold_winner else None,
        "evaluation_stage": "batch",
        "config_path": str(config_path),
        "snapshot_manifest": snapshot_manifest,
        "cadence_policy": settings["cadence"],
        "learning_policy": {
            "granularity": settings["learning_granularity"],
            "action_mode": settings["learning_action_mode"],
        },
        "challenge": challenge_stage,
        "gold": gold_stage,
        "selection": {
            "challenge_winner_family": challenge_winner["family"] if challenge_winner else None,
            "challenge_winner_interval": challenge_winner["interval"] if challenge_winner else None,
            "gold_winner_family": gold_winner["family"] if gold_winner else None,
            "gold_winner_interval": gold_winner["interval"] if gold_winner else None,
            "stable_winner": bool(
                challenge_winner
                and gold_winner
                and challenge_winner["family"] == gold_winner["family"]
            ),
            "baseline_family": settings["baseline_family"],
        },
        "decision_summary": (
            f"Batch {settings['batch_id']} ranks "
            f"{gold_winner['family'] if gold_winner else 'none'} as the gold-stage leader."
        ),
        "lesson_summary": (
            "Keep the baseline unless the gold winner also validates in demo."
            if gold_winner and gold_winner["family"] != settings["baseline_family"]
            else "The baseline remains the safest current production reference."
        ),
    }

    demo_followup = build_demo_followup(
        batch_id=settings["batch_id"],
        snapshot_id=settings["snapshot_id"],
        expected_candidate=gold_winner,
    )
    league_audit = build_league_audit(
        config=config,
        batch_id=settings["batch_id"],
        snapshot_id=settings["snapshot_id"],
        challenge_stage=challenge_stage,
        gold_stage=gold_stage,
        demo_followup=demo_followup,
    )
    project_learning = build_project_learning(
        config=config,
        batch_id=settings["batch_id"],
        snapshot_id=settings["snapshot_id"],
        batch_root=batch_output_root,
        gold_stage=gold_stage,
        league_audit=league_audit,
    )
    decision_review = build_decision_review(
        batch_id=settings["batch_id"],
        snapshot_id=settings["snapshot_id"],
        challenge_stage=challenge_stage,
        gold_stage=gold_stage,
        league_audit=league_audit,
        demo_followup=demo_followup,
        project_learning=project_learning,
    )

    outputs = {
        "league_leaderboard": save_json(
            batch_dir / LEAGUE_LEADERBOARD_FILENAME, league_leaderboard
        ),
        "league_audit": save_json(batch_dir / LEAGUE_AUDIT_FILENAME, league_audit),
        "demo_followup": save_json(batch_dir / LEAGUE_DEMO_FOLLOWUP_FILENAME, demo_followup),
        "decision_review": save_json(batch_dir / LEAGUE_DECISION_REVIEW_FILENAME, decision_review),
        "project_learning": save_json(
            batch_dir / LEAGUE_PROJECT_LEARNING_FILENAME, project_learning
        ),
    }
    return {
        "batch_id": settings["batch_id"],
        "snapshot_id": settings["snapshot_id"],
        "output_dir": str(batch_dir),
        "selection": league_leaderboard["selection"],
        "audit_verdict": league_audit["verdict"],
        "demo_status": demo_followup["status"],
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
