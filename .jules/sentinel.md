## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-07 - Missing Authentication Check
**Vulnerability:** Missing Authentication on `/api` endpoints in `dashboard/app.py`.
**Learning:** Adding `@app.before_request` hook provides robust, central protection to all endpoints instead of adding it manually to each route.
**Prevention:** Start projects with a central authentication/authorization scheme and default to deny. Use environment variables like `DASHBOARD_USERNAME` and `DASHBOARD_PASSWORD`.
