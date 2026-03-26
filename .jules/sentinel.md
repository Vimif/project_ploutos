## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2026-03-26 - Overly Permissive CORS Configuration
**Vulnerability:** The Flask application `dashboard/app.py` was configured with `CORS(app)` and `SocketIO(app, cors_allowed_origins="*")`, which allows any website to make requests to the API and WebSocket endpoints.
**Learning:** Default configurations for CORS and SocketIO in Flask often allow all origins by default or via the `"*"` wildcard, exposing the application to Cross-Site Request Forgery (CSRF) and unauthorized data access.
**Prevention:** Always restrict CORS and WebSocket origins to known, trusted domains using an environment variable (e.g., `ALLOWED_ORIGINS`). For REST APIs, restrict the CORS scope to specific routes (e.g., `r'/api/*'` instead of the entire app).
