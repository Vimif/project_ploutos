## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-23 - Unauthenticated Dashboard Access
**Vulnerability:** The Flask dashboard (`dashboard/app.py`) exposed sensitive trading endpoints (e.g., `/api/account`, `/api/positions`, `/api/close_position`) and data without any authentication, allowing unauthorized users to view portfolio data and close positions.
**Learning:** Internal dashboards or experimental UIs often lack authentication initially to simplify development, but can become a critical risk if exposed or deployed.
**Prevention:** Always implement basic authentication or identity verification by default on any web interface that exposes sensitive actions or data, even in early development or "paper trading" modes.
