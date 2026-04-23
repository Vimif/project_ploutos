"""Read-only Flask dashboard for monitoring the eToro demo session."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

from dashboard.demo_monitor import DemoMonitorService

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
allowed_origins = os.getenv("ALLOWED_CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000,http://localhost:3000").split(",")
CORS(app, resources={r"/*": {"origins": allowed_origins}})

demo_service = DemoMonitorService()


@app.context_processor
def inject_now() -> dict[str, Any]:
    from datetime import datetime

    return {"now": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def _serialize_session(session) -> dict[str, Any]:
    if not session:
        return {
            "session_id": None,
            "session_dir": None,
            "meta": {},
            "report": {},
            "latest_event_at": None,
            "latest_equity_at": None,
        }

    latest_event_at = session.events[-1].get("timestamp") if session.events else None
    latest_equity_at = session.equity[-1].get("timestamp") if session.equity else None
    return {
        "session_id": session.session_id,
        "session_dir": str(session.session_dir),
        "meta": session.meta,
        "report": session.report,
        "latest_event_at": latest_event_at,
        "latest_equity_at": latest_equity_at,
    }


def _build_overview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload["session"]
    broker = payload["broker"]
    diagnostics = payload["diagnostics"]
    league = payload["league_context"]

    latest_equity = session.equity[-1] if session and session.equity else {}
    account = broker["account"] if broker.get("connected") else {
        "portfolio_value": latest_equity.get("equity", 0.0),
        "cash": latest_equity.get("balance", 0.0),
        "equity": latest_equity.get("equity", 0.0),
    }
    return {
        "session": _serialize_session(session),
        "broker": broker,
        "account": account,
        "positions": broker.get("positions", []),
        "open_orders": broker.get("open_orders", []),
        "alerts": diagnostics["alerts"],
        "summary": session.report.get("summary", {}) if session else {},
        "league": league,
        "data_health": {
            "broker_connected": broker.get("connected", False),
            "broker_error": broker.get("error"),
            "desync": diagnostics.get("desync"),
        },
    }


def _timeline_payload(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload["session"]
    if not session:
        return {"session": _serialize_session(None), "events": []}
    limit = int(request.args.get("limit", 200))
    events = list(reversed(session.events[-limit:]))
    return {"session": _serialize_session(session), "events": events}


def _equity_payload(payload: dict[str, Any]) -> dict[str, Any]:
    session = payload["session"]
    return {
        "session": _serialize_session(session),
        "equity_curve": session.equity if session else [],
    }


@app.route("/")
@app.route("/demo")
@app.route("/demo/overview")
def demo_overview():
    return render_template("demo_overview.html", page="overview")


@app.route("/trades")
@app.route("/demo/session")
def demo_session():
    return render_template("demo_session.html", page="session")


@app.route("/metrics")
@app.route("/demo/insights")
def demo_insights():
    return render_template("demo_insights.html", page="insights")


@app.route("/api/demo/overview")
def api_demo_overview():
    payload = demo_service.get_demo_payload()
    return jsonify({"success": True, "data": _build_overview_payload(payload)})


@app.route("/api/demo/timeline")
def api_demo_timeline():
    payload = demo_service.get_demo_payload()
    return jsonify({"success": True, "data": _timeline_payload(payload)})


@app.route("/api/demo/equity")
def api_demo_equity():
    payload = demo_service.get_demo_payload()
    return jsonify({"success": True, "data": _equity_payload(payload)})


@app.route("/api/demo/diagnostics")
def api_demo_diagnostics():
    payload = demo_service.get_demo_payload()
    return jsonify(
        {
            "success": True,
            "data": {
                "session": _serialize_session(payload["session"]),
                "diagnostics": payload["diagnostics"],
            },
        }
    )


@app.route("/api/demo/recommendations")
def api_demo_recommendations():
    payload = demo_service.get_demo_payload()
    return jsonify(
        {
            "success": True,
            "data": {
                "session": _serialize_session(payload["session"]),
                "recommendations": payload["recommendations"],
            },
        }
    )


@app.route("/api/demo/historical-context")
def api_demo_historical_context():
    payload = demo_service.get_demo_payload()
    return jsonify({"success": True, "data": payload["historical_context"]})


@app.route("/api/demo/league")
def api_demo_league():
    payload = demo_service.get_demo_payload()
    return jsonify({"success": True, "data": payload["league_context"]})


@app.route("/api/demo/project-learning")
def api_demo_project_learning():
    payload = demo_service.get_demo_payload()
    return jsonify({"success": True, "data": payload["project_learning"]})


@app.route("/api/health")
def api_health():
    payload = demo_service.get_demo_payload()
    return jsonify(
        {
            "success": True,
            "status": "healthy",
            "session_available": payload["session"] is not None,
            "broker_connected": payload["broker"].get("connected", False),
        }
    )


@app.route("/api/db/trades")
def api_db_trades():
    payload = demo_service.get_demo_payload()
    session = payload["session"]
    trades = []
    if session:
        for event in session.events:
            if event.get("type") != "trade":
                continue
            trades.append(
                {
                    "timestamp": event.get("timestamp"),
                    "symbol": event.get("symbol") or event.get("ticker"),
                    "action": event.get("action") or event.get("side"),
                    "quantity": event.get("quantity", event.get("qty")),
                    "price": event.get("price", 0.0),
                    "amount": event.get("amount", event.get("total_value", 0.0)),
                    "reason": event.get("reason", ""),
                }
            )
    return jsonify({"success": True, "data": list(reversed(trades)), "count": len(trades), "source": "session_jsonl"})


@app.route("/api/db/evolution")
def api_db_evolution():
    payload = demo_service.get_demo_payload()
    session = payload["session"]
    data = []
    if session:
        for point in session.equity:
            data.append(
                {
                    "date": point.get("timestamp", "")[:10],
                    "timestamp": point.get("timestamp"),
                    "portfolio_value": point.get("equity"),
                    "balance": point.get("balance"),
                    "drawdown": point.get("drawdown"),
                    "exposure": point.get("exposure"),
                    "trades": None,
                    "volume": None,
                }
            )
    return jsonify({"success": True, "data": data, "source": "session_jsonl"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
