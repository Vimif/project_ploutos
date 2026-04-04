## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2024-04-04 - Hardcoded Basic Auth Credentials in Setup Scripts
**Vulnerability:** Found `admin:admin` hardcoded in URLs (`http://admin:admin@localhost:3000`) for Grafana API requests in setup scripts.
**Learning:** Basic Auth credentials should not be hardcoded in URLs as they could leak in logs and script files. Python's `requests` library accepts a tuple for the `auth` parameter, and `curl` has the `-u` option to provide basic auth securely.
**Prevention:** Use environment variables (via `python-dotenv` for Python and `${VAR:-default}` for Bash scripts) to inject these credentials securely, passing them via the respective parameter bindings instead of appending directly to the URL.
