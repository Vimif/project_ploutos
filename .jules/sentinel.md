## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2026-04-22 - Overly Permissive CORS Configuration
**Vulnerability:** The Flask application `dashboard/app.py` was initialized with `CORS(app)` without restrictive configurations, meaning it accepted cross-origin requests from any domain by default, potentially opening up vectors for CSRF or data leakage.
**Learning:** `flask-cors` configuration defaults to wildcards (`*`). While convenient for prototyping, this default behavior must be explicitly overridden for secure deployments.
**Prevention:** Always define explicit CORS origins and utilize `os.getenv` to inject a comma-separated list of safe domains for specific environments, providing local environments with localhost defaults to prevent development friction.
