## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-23 - Type-Safety Bypass in Parameterized Database Queries
**Vulnerability:** psycopg2 parameterized queries using string interpolation for `INTERVAL` like `INTERVAL %s` with parameters like `f'{days} days'` bypassed application-layer type safety checks. This left the application vulnerable to subtle injection or unexpected behaviour if `days` derived from user input.
**Learning:** Using psycopg2's variable substitution to build partial strings (e.g. `1 day`) avoids strict SQL injection by escaping input, but it doesn't enforce Python-level type safety. Passing an uncast string into the database bypasses application-layer defenses.
**Prevention:** Strictly cast input parameters (e.g., `int(days)`) before passing them to parameterized queries. When dealing with Postgres INTERVALs, use `INTERVAL '1 day' * %s` to enable strict numeric type-checking.
