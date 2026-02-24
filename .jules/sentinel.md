## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-23 - Unprotected Dashboard Endpoints
**Vulnerability:** The dashboard (`dashboard/app.py`) exposed sensitive financial data and trading actions (e.g., `close_position`) without any authentication.
**Learning:** The application was likely intended for local development or internal use, but exposed sensitive functionality via web routes that could be accessed if the network is compromised.
**Prevention:** Always implement authentication on web interfaces, even for internal tools. Use `@before_request` to apply security policies globally to avoid missing new endpoints.
