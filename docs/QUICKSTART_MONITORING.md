# Monitoring Quickstart

## 1. Install the web dependencies

```bash
pip install -e ".[web]"
```

## 2. Start the local dashboard

```bash
python dashboard/app.py
```

Open:

- `http://127.0.0.1:5000/demo/overview`
- `http://127.0.0.1:5000/demo/session`
- `http://127.0.0.1:5000/demo/insights`

## 3. Produce a demo session

```bash
python scripts/paper_trade.py --mode etoro --config config/config.yaml
```

If you do not want to hit the broker while testing the UI:

```bash
python scripts/paper_trade.py --mode simulate --config config/config.yaml
```

## 4. Check health

```bash
curl http://127.0.0.1:5000/api/health
```

## 5. Review the important API payloads

```bash
curl http://127.0.0.1:5000/api/demo/overview
curl http://127.0.0.1:5000/api/demo/recommendations
curl http://127.0.0.1:5000/api/demo/project-learning
```

## What You Should See

- current account or simulated equity
- recent trades and rejections
- risk alerts
- latest strategy or league context
- recommendations grounded in both the live session and historical audits

## If No Session Appears

- make sure `scripts/paper_trade.py` has written a session under `logs/paper_trading/`
- check `http://127.0.0.1:5000/api/health`
- inspect:
  - `logs/dashboard_stdout.log`
  - `logs/dashboard_stderr.log`
  - the latest session `report.json`

## Historical Note

Older Grafana or Streamlit monitoring instructions are no longer current.
Those assets now live under `legacy/`.
