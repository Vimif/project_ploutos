## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2024-05-18 - Overly Permissive CORS Configuration
**Vulnerability:** The Flask dashboard application (`dashboard/app.py`) was configured with `CORS(app)`, which defaults to allowing cross-origin requests from any origin (`*`).
**Learning:** This exposes the application to Cross-Origin Resource Sharing (CORS) attacks, where unauthorized domains can make requests to the dashboard. Although it is read-only, it is best practice to restrict CORS.
**Prevention:** Always restrict CORS `origins` in Flask applications. Use an environment variable to define allowed origins, falling back to local development ports (e.g., `localhost:5000`) rather than defaulting to `*`.
