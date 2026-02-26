## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-26 - Unauthenticated Sensitive Dashboard Actions
**Vulnerability:** The dashboard API (`dashboard/app.py`) exposed sensitive endpoints like `/api/close_position` without any authentication, allowing anyone with network access to manipulate the trading portfolio.
**Learning:** Default Flask app templates often lack authentication middleware. Critical actions (like closing positions) must always be protected by at least Basic Auth or API keys, even in "internal" dashboards.
**Prevention:** Implement a `before_request` hook or middleware that enforces authentication on all non-static routes by default. Use `secrets.compare_digest` for secure password comparison.
