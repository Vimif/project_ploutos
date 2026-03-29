## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2024-05-18 - Parameterized Queries Bypassing Type Safety
**Vulnerability:** SQL injection risks due to using `INTERVAL %s` with string-formatted arguments (e.g., `f"{days} days"`) and uncast user inputs for `LIMIT %s`. While `psycopg2` parameterized queries prevent traditional syntax breaking by escaping values, passing dynamically formatted strings bypassing application-level type safety can still lead to logic abuse.
**Learning:** Merely escaping strings is insufficient for SQL structural components like `INTERVAL`. If an input is purely numerical, it must be explicitly cast to an integer at the application level before being sent to the database.
**Prevention:** Explicitly cast numerical parameters to `int()` (or appropriate types) within `try...except` blocks and reconstruct SQL queries to rely on database-level mathematical transformations instead of string formatting, such as substituting `INTERVAL %s` with `INTERVAL '1 day' * %s` and passing pure integers.
