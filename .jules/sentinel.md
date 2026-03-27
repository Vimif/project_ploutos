## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2024-05-30 - Overly Permissive CORS and SocketIO Origins
**Vulnerability:** The Flask application and SocketIO instance in `dashboard/app.py` were using `CORS(app)` and `cors_allowed_origins="*"` without any restrictions, allowing any domain to make cross-origin requests to the API and WebSocket server.
**Learning:** This existed because permissive CORS headers are often used during initial development for convenience, bypassing browser same-origin policies.
**Prevention:** Define a restricted list of allowed origins via the `ALLOWED_ORIGINS` environment variable (e.g., `http://localhost:5000`) and apply CORS exclusively to the specific API routes (e.g., `r'/api/*'`) that need it.
