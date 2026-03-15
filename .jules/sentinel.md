## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2026-03-15 - Missing Application Endpoint Authentication
**Vulnerability:** The main Flask dashboard application (`dashboard/app.py`) and its WebSocket connections completely lacked authentication, meaning any user could view portfolio metrics and execute trades manually (e.g., `/api/close_position`).
**Learning:** Initial dashboard iterations (like this JSON-based mode) are often built to just work locally but expose critical endpoints if deployed externally. Using Flask decorators (`@app.before_request`) ensures authentication applies automatically globally without modifying every route.
**Prevention:** Always implement "secure by default" patterns. If authentication keys aren't provided via environment variables, the application should strictly deny access rather than falling back to an open or default state. Require explicit `TESTING` flags to bypass auth in test environments to prevent accidental circumvention.
