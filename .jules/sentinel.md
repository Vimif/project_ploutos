## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-28 - Uncast PostgreSQL Interval SQL Injection Risk
**Vulnerability:** SQL queries in `database/db.py` used string concatenation (like `INTERVAL %s` passing `f'{days} days'`) and uncast variable replacements for PostgreSQL queries (e.g. `LIMIT %s` passing uncast `limit`), which bypassed Python-level type safety.
**Learning:** Even though `psycopg2` parameterized queries prevent strict SQL injection by escaping values, passing string-formatted arguments or uncast variables into structural keywords like `INTERVAL` or `LIMIT` leaves the query vulnerable to unexpected input types or application-level errors.
**Prevention:** For defense-in-depth, strictly cast user inputs (e.g., `int(days)`, `int(limit)`) before passing them to the database driver, and prefer parameterized arithmetic in SQL like `INTERVAL '1 day' * %s` to enforce type constraints at the database layer.
