## 2024-05-24 - Default Database Credentials Hardcoded
**Vulnerability:** The PostgreSQL database credentials (ploutos/your_password) were hardcoded in the README and potentially default fallback code.
**Learning:** Default credentials left in code or configuration files are a common entry point for attackers if the service is accidentally exposed.
**Prevention:** Always require database credentials to be explicitly set via environment variables (e.g., `DB_PASSWORD`) and fail securely if they are missing.

## 2024-05-24 - Missing Authentication on Dashboard
**Vulnerability:** The Ploutos dashboard exposed sensitive financial and operational endpoints (like `/api/account`, `/api/positions`, `/api/close_position/<symbol>`) without any authentication mechanism.
**Learning:** Hardcoded default credentials or no authentication can easily leak financial data or allow unauthorized actions.
**Prevention:** Implement globally-enforced HTTP Basic Authentication (`app.before_request`) using securely configured environment variables, while ensuring the application fails securely (denies access) when these variables are missing.
