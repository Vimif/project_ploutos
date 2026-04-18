## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2026-04-18 - Restrict network exposure and permissive CORS
**Vulnerability:** The Flask dashboard (`dashboard/app.py`) was binding to `0.0.0.0` allowing connections from any IP, and `flask_cors` was initialized with `CORS(app)` allowing any origin (`*`) to query it.
**Learning:** These settings make local development simple, but are overly permissive and expose the application in production or shared network scenarios. It is better to fail securely by restricting the service to localhost by default.
**Prevention:** Never bind to `0.0.0.0` or use wildcard CORS unless explicitly needed via configuration. Always use environment variables (e.g. `DASHBOARD_HOST` and `ALLOWED_CORS_ORIGINS`) to handle network bindings with secure local defaults.
