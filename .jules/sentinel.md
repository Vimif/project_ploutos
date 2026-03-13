## 2026-02-23 - Duplicated Flask Configuration Vulnerability
**Vulnerability:** Hardcoded `SECRET_KEY` found in multiple dashboard applications (`dashboard/app.py` and `dashboard/app_v2.py`).
**Learning:** The vulnerability was propagated because `app_v2.py` was likely created by copying `app.py` without reviewing security configurations.
**Prevention:** When versioning applications (v1 -> v2), perform a security diff or review configuration files to ensure secrets are not carried over. Use environment variables for all sensitive configuration from the start.

## 2026-03-13 - CI Mock Test Leakage Vulnerability
**Vulnerability:** The Pytest global mock for `torch` using `sys.modules.setdefault("torch", MagicMock())` at module level leaked out to `test_full_pipeline_execution`, breaking standard instantiation of PPO networks with obscure exceptions (`TypeError: isinstance() arg 2 must be a type`).
**Learning:** Pytest does not completely sandbox execution state when multiple test suites import and cache shared dependencies containing global sys.modules hacks, introducing unreliable flakey builds and preventing actual validation of pipeline dependencies.
**Prevention:** Avoid mutating global namespaces permanently. Use try-except fallback imports (`try: import torch; except ImportError: sys.modules["torch"] = MagicMock()`) so that real environments correctly override mocks.
