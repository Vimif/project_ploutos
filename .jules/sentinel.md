## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.
## 2025-02-23 - Default Network Exposure Risk
**Vulnerability:** Flask application in `dashboard/app.py` was bound to `0.0.0.0` by default, exposing the local development server to all network interfaces.
**Learning:** Hardcoded `0.0.0.0` bindings unnecessarily widen the attack surface, allowing potential unauthorized access from other machines on the network if the firewall is misconfigured. Containerized deployments (like Docker) often require `0.0.0.0` to route traffic properly.
**Prevention:** Default to local loopback `127.0.0.1` binding to minimize network exposure. Allow the host to be overriden via an environment variable (e.g., `os.environ.get("HOST", "127.0.0.1")`) to support flexible deployment strategies like containerization without compromising secure defaults.
