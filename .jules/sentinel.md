## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2025-10-24 - Missing Authentication on Dashboard

**Vulnerability:** The trading dashboard did not require authentication, exposing sensitive trading data and manual control mechanisms to anyone with access to the URL.
**Learning:** The Flask application didn't enforce a global authentication check. Because it exposes trading metrics, API status, and allows manual position closures (like `/api/close_position/<symbol>`), missing authorization allows critical control and data leakage.
**Prevention:** Implementing a global `before_request` HTTP Basic Auth check with explicit exclusions (like `/static` or `OPTIONS` requests) ensures all routes are secure by default.
