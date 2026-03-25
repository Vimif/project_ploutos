## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2026-02-23 - Defense-in-depth for parameterized SQL queries
**Vulnerability:** Weakness in SQL queries where type-unsafe input variables (`days`, `limit`) were passed dynamically via string-formatting (`INTERVAL %s`) instead of using explicit mathematical calculations. This allows attackers or user errors to bypass Python-level type-safety.
**Learning:** `psycopg2` strictly escapes execution variables to prevent raw SQL injections, however, `INTERVAL %s` using formatted strings (`f'{days} days'`) or variables uncast in `LIMIT %s` bypass Python-level type safety, making applications vulnerable to application logic bugs.
**Prevention:** Always enforce strict type casting (`int()`, `float()`) for mathematical inputs and limits in SQL queries. Avoid dynamic string formatting within the query definition, explicitly using syntax like `INTERVAL '1 day' * %s` to validate constraints at the application layer.
