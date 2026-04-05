# Monitoring

The supported monitoring surface for Ploutos is the local Flask dashboard under
`dashboard/`.

It is centered on the eToro demo session and stays read-only in V1.

## Current Monitoring Stack

- dashboard server: `python dashboard/app.py`
- live session source: `logs/paper_trading/<session_id>/`
- broker truth source: eToro demo account through the live broker adapter
- historical context:
  - `logs/strategy_compare/**/strategy_leaderboard.json`
  - `logs/profitability/*.json`
  - `logs/league_batches/**/league_leaderboard.json`
  - `logs/league_batches/**/project_learning.json`

The old Grafana or Streamlit monitoring stack has been archived under `legacy/`
and is no longer the supported path.

## Pages

- `/demo/overview`
- `/demo/session`
- `/demo/insights`

## API Endpoints

- `GET /api/demo/overview`
- `GET /api/demo/timeline`
- `GET /api/demo/equity`
- `GET /api/demo/diagnostics`
- `GET /api/demo/recommendations`
- `GET /api/demo/historical-context`
- `GET /api/demo/league`
- `GET /api/demo/project-learning`
- `GET /api/health`

Compatibility endpoints backed by the same session files also remain available:

- `GET /api/db/trades`
- `GET /api/db/evolution`

## Canonical Session Files

Each demo session writes to:

- `session_meta.json`
- `events.jsonl`
- `equity.jsonl`
- `report.json`
- `journal.json` for legacy export compatibility

The dashboard treats these files as the main local explanation layer.

## Refresh Behavior

- broker cache: 15 seconds
- UI polling: 5 seconds
- mode: read-only

## Alert Types

The dashboard highlights:

- drawdown near limit
- daily loss near limit
- broker/journal desync
- price unavailable
- duplicate signal blocked
- observation mismatch
- repeated rejection spikes

## Improvement Insights

The dashboard does not just show live state. It also blends:

- current session diagnostics
- the latest profitability audit
- the latest strategy comparison winner
- the latest league winner
- project-learning history alerts and good or bad patterns

This is the recommended surface for deciding what to improve next.
