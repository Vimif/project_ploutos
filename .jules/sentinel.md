## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2024-05-19 - PostgreSQL INTERVAL Parameter Injection Prevention
**Vulnerability:** Constructing PostgreSQL INTERVAL arguments using Python string formatting (e.g., `f"{days} days"`) and passing them directly into `INTERVAL %s` parameters is susceptible to SQL injection and input manipulation.
**Learning:** Even when the original source logic appears to just be an integer `days`, interpolating values into query substrings can be exploited or cause query syntax failures.
**Prevention:** Always use safe multiplier syntax like `INTERVAL '1 day' * %s` and pass the numeric value strictly as an integer (e.g., `int(days)`) in the query execution parameters.
