## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2024-05-24 - Missing Authentication on Dashboard
**Vulnerability:** The Ploutos dashboard exposed sensitive financial and operational endpoints (like `/api/account`, `/api/positions`, `/api/close_position/<symbol>`) without any authentication mechanism.
**Learning:** Hardcoded default credentials or no authentication can easily leak financial data or allow unauthorized actions.
**Prevention:** Implement globally-enforced HTTP Basic Authentication (`app.before_request`) using securely configured environment variables, while ensuring the application fails securely (denies access) when these variables are missing.
