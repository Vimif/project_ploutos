## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-24 - Unauthenticated Dashboard Exposure
**Vulnerability:** The Flask dashboard (`dashboard/app.py`) exposed sensitive financial data and trading controls (`POST /api/close_position`) without any authentication.
**Learning:** Development tools (like dashboards) often start as "local-only" but can be inadvertently exposed.
**Prevention:** Default to "secure by default" for all web interfaces, even internal ones. Use a simple Basic Auth decorator as a minimum baseline before any deployment.
