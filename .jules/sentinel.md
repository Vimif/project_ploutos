## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-03 - Missing Dashboard Authentication
**Vulnerability:** The main dashboard application (`dashboard/app.py`) was completely exposed without any authentication, allowing unauthorized access to sensitive financial data, trading history, and portfolio controls.
**Learning:** Web dashboards built for internal use or algorithmic trading bots often neglect authentication under the assumption they will be deployed locally or protected by a reverse proxy. However, this creates a significant security gap if exposed directly or if local access is compromised.
**Prevention:** Always implement an authentication layer (like HTTP Basic Auth) directly within the application code (`app.before_request`) as a defense-in-depth measure, even if external protections are planned. Use `secrets.compare_digest` to prevent timing attacks when comparing credentials.
