## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2023-11-20 - Overly Permissive CORS and WebSocket Origins
**Vulnerability:** The Flask application (`dashboard/app.py`) configured `flask_cors.CORS` and `flask_socketio.SocketIO` to allow any origin (`*`). This misconfiguration allows malicious websites to make cross-origin REST API and WebSocket requests using the authenticated user's session, leading to Cross-Site Request Forgery (CSRF) and Cross-Origin Resource Sharing security risks.
**Learning:** Hardcoded `*` parameters in CORS and WebSockets are overly permissive default configurations used during development that bypass critical browser security mechanisms in production.
**Prevention:** Always restrict `CORS` origins and `cors_allowed_origins` for `SocketIO` strictly to the domains that require access, using environment variables like `ALLOWED_ORIGINS` to specify comma-separated trusted domains. Apply CORS specifically to the `/api/*` resources rather than globally.
