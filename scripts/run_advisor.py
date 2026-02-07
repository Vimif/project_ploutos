#!/usr/bin/env python3
"""
Lancer le dashboard Ploutos Advisory.

Usage:
    python scripts/run_advisor.py
    python scripts/run_advisor.py --port 5001
    python scripts/run_advisor.py --host 0.0.0.0 --port 8080
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ploutos Advisory Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host (defaut: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001, help="Port (defaut: 5001)")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  PLOUTOS ADVISORY DASHBOARD")
    logger.info("=" * 60)
    logger.info(f"  URL: http://{args.host}:{args.port}")
    logger.info(f"  Debug: {args.debug}")
    logger.info("=" * 60)

    from web_advisor.app import create_app

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
