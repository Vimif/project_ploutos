## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2025-05-18 - Overly Permissive CORS Configuration
**Vulnerability:** `CORS(app)` without restrictive `resources` or `origins` allows arbitrary domains to make requests.
**Learning:** Default Flask-CORS behavior allows requests from all origins (`*`), leaving APIs vulnerable if session-based auth or cookies are somehow enabled or if local services bind implicitly.
**Prevention:** Always restrict CORS using dynamic, environment-based allowed origins with safe fallbacks.
