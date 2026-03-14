## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-14 - PostgreSQL INTERVAL Parameter Injection Fix
**Vulnerability:** PostgreSQL `INTERVAL` queries in `database/db.py` used string concatenation (e.g., `(f'{days} days', )`) to build interval parameters for SQL queries. Though partially sanitized by `psycopg2` via the `%s` placeholder, constructing query segments with string interpolation before parameter binding could risk type confusion or bypass if strict `int` typings fail or are omitted, paving the way for injection or query malformations.
**Learning:** `psycopg2` requires strictly typed parameter passing and avoids string manipulation for database logic like `INTERVAL`.
**Prevention:** Avoid constructing strings for `INTERVAL` like `f'{days} days'`. Use parameter multiplication `INTERVAL '1 day' * %s` and cast input variables explicitly to `int` within query arguments to guarantee type safety and prevent injection vulnerabilities.
