## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-23 - Insecure Date Intervals and Limit Parameterization in SQL Queries
**Vulnerability:** Weakly typed SQL parameterization bypassing `psycopg2` escaping for INTERVAL and LIMIT statements, exposing queries to potential application-level injection and bugs in `database/db.py`.
**Learning:** `psycopg2` parameterized queries prevent strict SQL injection by escaping values, but passing string-formatted arguments like `f'{days} days'` into `INTERVAL %s` or uncast variables into `LIMIT %s` bypasses Python-level type safety.
**Prevention:** For defense-in-depth, strictly cast inputs (e.g., `int(days)`) and use syntax like `INTERVAL '1 day' * %s` to validate at the application layer.
