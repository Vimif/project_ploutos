## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2024-05-20 - Unrestricted CORS Allow-All By Default
**Vulnerability:** The Flask application `dashboard/app.py` was initialized with `CORS(app)`, which defaults to allowing all origins (`*`), creating a cross-origin security gap where unauthorized sites could potentially read API data if auth was ever attached or local resources were targeted.
**Learning:** `flask-cors` behaves permissively by default. In internal or dashboard applications, this can easily lead to unintended exposure, especially on `localhost` bindings.
**Prevention:** Always restrict `flask-cors` explicitly using the `resources` dictionary, tying `origins` to a configurable environment variable with safe, restrictive local defaults (e.g., `localhost` specific ports).
