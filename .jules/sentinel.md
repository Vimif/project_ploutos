## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2023-10-27 - Fail Secure Authentication Strategy
**Vulnerability:** The dashboard application completely lacked an authentication layer, allowing unrestricted public access to sensitive administrative endpoints like `/api/close_position/<symbol>`, which constitutes a CRITICAL security risk (Broken Access Control).
**Learning:** When adding HTTP Basic Auth to an existing unauthenticated dashboard via `@app.before_request`, relying on hardcoded defaults like `admin/admin` as fallbacks for missing environment variables creates a new vulnerability. If the variables are not set during deployment, the system remains exposed with publicly known credentials.
**Prevention:** Always implement a "fail secure" strategy. If necessary security configuration (like `DASHBOARD_USERNAME` or `DASHBOARD_PASSWORD`) is missing, actively deny access and fail securely rather than attempting to self-heal with hardcoded fallbacks.
