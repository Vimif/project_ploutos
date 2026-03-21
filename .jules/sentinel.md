## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-14 - PostgreSQL Input Validation for INTERVAL
**Vulnerability:** While `psycopg2` parameterized queries prevent strict SQL injection by escaping values, passing string-formatted arguments like `f'{days} days'` into `INTERVAL %s` or uncast variables into `LIMIT %s` bypasses Python-level type safety. This relies entirely on the database engine to catch malformed strings.
**Learning:** Defense-in-depth requires validating and casting inputs at the application layer before passing them to the database, even when using parameterized queries.
**Prevention:** To secure PostgreSQL `INTERVAL` queries and bounds, use `INTERVAL '1 day' * %s` and cast parameters to their required numeric type (e.g., `int(days)`, `int(limit)`). This ensures strict type safety and fails fast on invalid input.
