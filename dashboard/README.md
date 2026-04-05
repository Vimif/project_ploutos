# Ploutos Demo Dashboard

Local Flask dashboard for monitoring the current demo session in read-only mode.

This is the supported dashboard path for the repository.

The older PostgreSQL, Grafana, and Streamlit material has been archived and is
no longer the mainline contract.

## Run

```bash
pip install -e ".[web]"
python dashboard/app.py
```

Open:

- `http://127.0.0.1:5000/demo/overview`
- `http://127.0.0.1:5000/demo/session`
- `http://127.0.0.1:5000/demo/insights`

## Data Sources

- latest session under `logs/paper_trading/<session_id>/`
- broker snapshot through the live broker adapter
- latest strategy comparison artifacts
- latest profitability audit
- latest league batch outputs

## API

Primary demo endpoints:

- `/api/demo/overview`
- `/api/demo/timeline`
- `/api/demo/equity`
- `/api/demo/diagnostics`
- `/api/demo/recommendations`
- `/api/demo/historical-context`
- `/api/demo/league`
- `/api/demo/project-learning`
- `/api/health`

Compatibility endpoints backed by the same session files:

- `/api/db/trades`
- `/api/db/evolution`

## Session Contract

The dashboard expects:

- `session_meta.json`
- `events.jsonl`
- `equity.jsonl`
- `report.json`

These files are written by `scripts/paper_trade.py` and read through
`dashboard/demo_monitor.py`.

## Scope

The dashboard is intentionally read-only in V1:

- no manual order placement
- no manual position close
- no control plane for the bot

Its role is to monitor the current session, explain what happened, and surface
the next improvement priorities.
