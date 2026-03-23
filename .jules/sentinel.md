## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2026-02-23 - PostgreSQL Interval/Limit Parameter Type Safety
**Vulnerability:** String-formatted parameterizations (e.g. `f'{days} days'`) into `INTERVAL %s` and uncast inputs into `LIMIT %s` lack application-layer type validation, increasing injection risk via weak inputs, even when using parameterized queries.
**Learning:** `psycopg2` parameterized execution relies on escaping strings natively but doesn't inherently cast Python primitives to strictly typed inputs. By explicitly casting dynamic query inputs via `int()` or utilizing safe PostgreSQL type operations (`INTERVAL '1 day' * %s`), the application enforces an additional layer of constraint checking.
**Prevention:** Avoid string-formatted date/interval parameterization directly inside queries. Always cast primitive arguments (e.g., limit, days) to strong integers in Python before execution, and use strict SQL multipliers (`INTERVAL '1 unit' * %s`) to guarantee exact type bounds.
