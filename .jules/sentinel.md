## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-04-12 - Overly Permissive CORS Configuration
**Vulnerability:** The Flask application `dashboard/app.py` was using `CORS(app)` without restrictions, allowing all origins by default.
**Learning:** This is a common misconfiguration in Flask applications using `flask-cors`. While acceptable for pure local development tools, it presents a CSRF and data leakage risk if the dashboard is accidentally exposed or if users interact with malicious sites while running the dashboard locally.
**Prevention:** Always scope CORS configurations. If dynamic configuration is required for different environments, use environment variables (e.g., `ALLOWED_CORS_ORIGINS`) with secure defaults, and specify explicit endpoints or paths instead of wildcard resources if possible.
