"""
Ploutos Advisory Dashboard - Application Flask principale.

Lance sur le port 5001 (coexiste avec le dashboard existant sur 5000).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import logging
from flask import Flask, render_template
from flask_cors import CORS

from advisory.engine import AdvisoryEngine
from config.advisory_config import AdvisoryConfig
from web_advisor.api.advisor_routes import advisor_bp, init_engine
from web_advisor.api.market_routes import market_bp
from web_advisor.api.portfolio_routes import portfolio_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Factory pour creer l'application Flask."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET", "ploutos-advisor-secret")
    CORS(app)

    # Initialiser le moteur advisory
    config = AdvisoryConfig()
    engine = AdvisoryEngine(config=config)
    init_engine(engine)

    # Enregistrer les blueprints API
    app.register_blueprint(advisor_bp)
    app.register_blueprint(market_bp)
    app.register_blueprint(portfolio_bp)

    # Routes pages
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/analysis/<symbol>")
    def analysis_page(symbol):
        return render_template("analysis.html", symbol=symbol.upper())

    @app.route("/api/health")
    def health():
        return {"status": "ok", "llm_available": engine.explainer.is_available}

    # Init DB advisory
    try:
        from database.advisory_db import init_advisory_tables

        init_advisory_tables()
    except Exception as e:
        logger.warning(f"Init DB advisory echouee (non bloquant): {e}")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5001, debug=True)
