## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-05-02 - Default Host Binding Vulnerability
**Vulnerability:** Flask development server in `dashboard/app.py` bound to `0.0.0.0` by default.
**Learning:** Exposing the local dashboard to all network interfaces on `0.0.0.0` is a security risk for local development, allowing access across the local network without authentication. However, overriding via an environment variable is necessary for Docker container environments.
**Prevention:** Ensure development servers bind to loopback (`127.0.0.1`) by default, and use a configurable `HOST` environment variable to support containerized workflows safely.
