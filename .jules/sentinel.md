## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-24 - Missing Authentication on Admin Dashboard
**Vulnerability:** The dashboard API and UI (`dashboard/app.py`) had no authentication, exposing sensitive trading data and administrative actions (like closing positions).
**Learning:** Even internal or local dashboards need authentication if they expose sensitive operations or data, especially if deployed to environments where access isn't strictly controlled by network policies. Failing to secure default configurations allows direct access.
**Prevention:** Implement secure default authentication (e.g., HTTP Basic Auth) on all admin/dashboard endpoints. Use `secrets.compare_digest` to prevent timing attacks and ensure the application fails securely (deny access) if credentials are not configured in the environment.
