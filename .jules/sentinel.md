## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-05 - Dashboard Unauthenticated Access
**Vulnerability:** Flask dashboard API and web interfaces were fully exposed without authentication, allowing any visitor to view account information, positions, trades, and manually close positions.
**Learning:** This exposes the application to potentially significant financial risk since an attacker could close positions via the `/api/close_position/<symbol>` endpoint.
**Prevention:** Always implement basic authentication or an API key requirement on internal or admin-facing dashboards by default, checking access on every request globally via middleware like `@app.before_request`.
