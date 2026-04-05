## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2025-05-18 - Overly Permissive CORS Configuration
**Vulnerability:** The dashboard API (`dashboard/app.py`) was using `CORS(app)` without restrictions, allowing any origin to make cross-origin requests to sensitive endpoints.
**Learning:** Incomplete CORS configuration exposes APIs to Cross-Site Request Forgery (CSRF) and unwanted access. By default, `flask-cors` opens up all routes to all domains if not specified.
**Prevention:** Scope CORS configurations strictly to the necessary routes (e.g., `r"/api/*"`) and configure origins explicitly via environment variables (`ALLOWED_ORIGINS`) rather than using default wildcards.
