## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-12 - PostgreSQL INTERVAL Injection Risk
**Vulnerability:** Constructing `INTERVAL` queries in `database/db.py` using standard f-string interpolation (e.g., `f'{days} days'`) or rigid format bindings allowed potential injection if the `days` parameter wasn't strictly checked, and failed to leverage built-in type safety mechanisms.
**Learning:** Using `INTERVAL %s` passing a pre-formatted string is insecure and prone to SQL syntax errors or subtle injection if inputs are unvalidated.
**Prevention:** Explicitly cast numeric inputs (e.g., `int(days)`) before building queries. For PostgreSQL intervals, use the syntax `INTERVAL '1 day' * %s` with an integer parameter binding to ensure absolute type safety and eliminate string manipulation vulnerabilities.
