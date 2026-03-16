## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-02-23 - PostgreSQL INTERVAL SQL Injection Risk
**Vulnerability:** String interpolation used in PostgreSQL queries for `INTERVAL` types (e.g., `INTERVAL %s` with parameter `f'{days} days'`) in `database/db.py`.
**Learning:** Directly embedding potentially unsafe strings into structural SQL parts like `INTERVAL` literals, even when using parameterized queries (`%s`), is poor practice and can be vulnerable to injection or manipulation if the input (`days`) is not strictly controlled.
**Prevention:** Always use explicit type casting in Python (e.g., `days = int(days)`) before query execution and use the mathematical syntax `INTERVAL '1 day' * %s` passing the integer parameter directly to the database driver to ensure type safety.
