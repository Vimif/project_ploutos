## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2025-03-31 - Defense-in-depth against SQL Injection via Interval formatting
**Vulnerability:** String formatted parameters (`f'{days} days'`) were passed to PostgreSQL queries containing `INTERVAL %s` or used as `LIMIT %s` boundaries. Even with Psycopg2 escaping parameters, this breaks Python-level type safety.
**Learning:** Using untyped variable inputs directly inside `INTERVAL` parameterizations or as uncast `LIMIT` variables creates risks if input data points are ever dynamically altered upstream, leading to potential unexpected evaluations.
**Prevention:** Strictly cast arguments to types at query time (e.g., `int(days)`, `int(limit)`), and implement logical operations entirely at the SQL syntax level (e.g., replacing `INTERVAL %s` with `INTERVAL '1 day' * %s`) to prevent any uncast parameter injections.
