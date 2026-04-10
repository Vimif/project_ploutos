## 2025-02-14 - Overly Permissive Default CORS
**Vulnerability:** CORS in `dashboard/app.py` was initialized with default `CORS(app)` which allowed wildcard `*` access to all endpoints.
**Learning:** Overly permissive CORS might be configured to easily facilitate frontend-backend communication in early development, but lacks proper restriction using environment variables. In Flask-CORS, it's safer to configure wildcard roots `r'/*'` with environment fallbacks than hardcoding specific APIs.
**Prevention:** Explicitly limit origins through `ALLOWED_CORS_ORIGINS` environment variables, retaining `localhost` variants for dev, instead of using default initialization.
