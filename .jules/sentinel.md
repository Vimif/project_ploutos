## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-02 - Missing Authentication on Dashboard
**Vulnerability:** The Flask dashboard for the trading bot (`dashboard/app.py`) was entirely open without any authentication, exposing sensitive trading data and administrative actions to anyone who could access the port.
**Learning:** Tools or dashboards meant for local or internal use often lack authentication, under the assumption they won't be exposed. This violates "defense in depth" as any misconfiguration exposing the port leads directly to full compromise. Implementing a fail-secure Basic Auth using `secrets.compare_digest` prevents this.
**Prevention:** Always implement at least Basic Authentication for sensitive internal tools using environment variables, and ensure the logic fails securely (denies access) if the configuration is missing or invalid.
