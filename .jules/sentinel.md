## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-17 - SQL Injection Vulnerability via INTERVAL formatting
**Vulnerability:** Passing user-controlled values via python `f'{days} days'` into PostgreSQL `INTERVAL %s` parameters creates a risk of SQL injection and type casting issues if the input string is manipulated.
**Learning:** While `psycopg2` sanitizes `%s` parameters, interpolating external variables directly into formatted string definitions like intervals circumvents these protections and can cause unexpected database errors or potential injections.
**Prevention:** Explicitly cast external inputs to `int` immediately upon entry to the function. At the SQL level, use `INTERVAL '1 day' * %s` to safely multiply the parameter mathematically, instead of concatenating strings.