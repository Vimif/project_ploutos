## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-13 - Missing Authentication on Dashboard
**Vulnerability:** The Ploutos dashboard (`dashboard/app.py`) lacked any authentication, exposing sensitive financial data, trading history, and manual trade execution endpoints to anyone with access to the port.
**Learning:** Even internal or "local" tools require authentication to prevent unauthorized access, especially when they can execute financial transactions or expose sensitive data.
**Prevention:** Implement global authentication mechanisms (like HTTP Basic Auth with `secrets.compare_digest` for timing attack protection) early in the development lifecycle for any administrative or monitoring interface. Fail securely if credentials are not configured.
