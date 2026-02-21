## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-24 - Missing Auth on Dashboard
**Vulnerability:** The dashboard API endpoints (e.g., `/api/close_position`) were completely unauthenticated, allowing anyone with network access to close positions.
**Learning:** Even internal tools need basic authentication. Flask apps don't have built-in auth by default.
**Prevention:** Apply a `@requires_auth` decorator to all sensitive routes. Use `secrets.compare_digest` for password checking.
