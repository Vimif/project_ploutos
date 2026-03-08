## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-23 - Dashboard Authentication Bypass
**Vulnerability:** Missing authentication on sensitive endpoints in `dashboard/app.py`.
**Learning:** The dashboard exposed sensitive endpoints like `/api/account` and `/api/positions` to the public without requiring authentication. In a production environment, this could lead to unauthorized access and information disclosure. The issue was fixed by adding a `@app.before_request` hook to enforce basic authentication.
**Prevention:** Always secure all sensitive API endpoints with proper authentication checks. Ensure `before_request` hooks are used to enforce security policies globally across all endpoints.
