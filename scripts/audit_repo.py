#!/usr/bin/env python3
"""Repository consistency audit for workflow and packaging drift."""

from __future__ import annotations

import argparse
import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCRIPT_ENTRY_RE = re.compile(r'^\s*([A-Za-z0-9._-]+)\s*=\s*"([^"]+)"\s*$')
PYTHON_CMD_RE = re.compile(r"(?m)^\s*(?:python|python3)\s+([A-Za-z0-9_./\\-]+\.py)\b")
SHELL_CMD_RE = re.compile(r"(?m)^\s*\./([A-Za-z0-9_./\\-]+\.sh)\b")
VERSION_RE = re.compile(r"\bV(\d+(?:\.\d+)?)\b")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


@dataclass
class Finding:
    severity: str
    source: str
    message: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def parse_project_scripts(pyproject_path: Path) -> dict[str, str]:
    """Parse `[project.scripts]` without an external TOML dependency."""
    entries: dict[str, str] = {}
    in_section = False
    for raw_line in _read_text(pyproject_path).splitlines():
        line = raw_line.strip()
        if line.startswith("["):
            if in_section:
                break
            in_section = line == "[project.scripts]"
            continue
        if not in_section or not line or line.startswith("#"):
            continue
        match = SCRIPT_ENTRY_RE.match(line)
        if match:
            entries[match.group(1)] = match.group(2)
    return entries


def extract_command_paths(readme_path: Path) -> tuple[list[str], list[str]]:
    """Extract Python and shell script paths referenced in README commands."""
    content = _read_text(readme_path)
    return PYTHON_CMD_RE.findall(content), SHELL_CMD_RE.findall(content)


def resolve_console_script_target(project_root: Path, target: str) -> Finding | None:
    """Validate that a console-script target points to an existing module:function."""
    module_name, sep, attr_name = target.partition(":")
    module_path = project_root / Path(*module_name.split(".")).with_suffix(".py")
    if not module_path.exists():
        return Finding("ERROR", "pyproject.toml", f"Console script target missing module: {target}")

    if not sep:
        return None

    tree = ast.parse(_read_text(module_path), filename=str(module_path))
    defined_names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if attr_name not in defined_names:
        return Finding(
            "ERROR",
            str(module_path.relative_to(project_root)),
            f"Console script target missing symbol '{attr_name}' for {target}",
        )
    return None


def check_readme_command_paths(project_root: Path, readme_path: Path) -> list[Finding]:
    findings: list[Finding] = []
    python_paths, shell_paths = extract_command_paths(readme_path)
    for rel_path in python_paths + shell_paths:
        candidate = project_root / Path(rel_path.replace("\\", "/"))
        if not candidate.exists():
            findings.append(
                Finding(
                    "ERROR", "README.md", f"Documented command points to missing path: {rel_path}"
                )
            )
    return findings


def check_version_markers(project_root: Path) -> list[Finding]:
    tracked_paths = [
        Path("README.md"),
        Path("config/config.yaml"),
        Path("scripts/paper_trade.py"),
    ]
    versions: dict[str, str] = {}
    for rel_path in tracked_paths:
        path = project_root / rel_path
        if not path.exists():
            continue
        preamble = "\n".join(_read_text(path).splitlines()[:40])
        match = VERSION_RE.search(preamble)
        if match:
            versions[str(rel_path)] = match.group(1)

    if len(set(versions.values())) <= 1:
        return []

    pretty = ", ".join(f"{path} -> V{version}" for path, version in versions.items())
    return [
        Finding(
            "WARN",
            "version-drift",
            f"Primary workflow files advertise inconsistent versions: {pretty}",
        )
    ]


def check_soft_failed_typecheck(project_root: Path) -> list[Finding]:
    workflow_path = project_root / ".github/workflows/tests.yml"
    if not workflow_path.exists():
        return []
    if "|| true" not in _read_text(workflow_path):
        return []
    return [
        Finding(
            "WARN",
            ".github/workflows/tests.yml",
            "Typecheck workflow contains '|| true', which can hide real failures.",
        )
    ]


def check_pytest_config_duplication(project_root: Path) -> list[Finding]:
    pyproject_path = project_root / "pyproject.toml"
    pytest_ini_path = project_root / "pytest.ini"
    if not pyproject_path.exists() or not pytest_ini_path.exists():
        return []

    pyproject_content = _read_text(pyproject_path)
    if "[tool.pytest.ini_options]" not in pyproject_content:
        return []

    return [
        Finding(
            "WARN",
            "pyproject.toml",
            "Pytest configuration is duplicated in pyproject.toml and pytest.ini. "
            "Pytest will ignore the pyproject section while pytest.ini exists.",
        )
    ]


def check_windows_ascii_safety(project_root: Path) -> list[Finding]:
    tracked_paths = [
        Path("README.md"),
        Path("pyproject.toml"),
        Path("config/config.yaml"),
        Path("training/train.py"),
        Path("scripts/run_pipeline.py"),
        Path("scripts/robustness_tests.py"),
        Path("scripts/paper_trade.py"),
        Path("scripts/backtest_ultimate.py"),
        Path("scripts/audit_repo.py"),
        Path("docs/MONITORING.md"),
        Path("docs/QUICKSTART_MONITORING.md"),
        Path("dashboard/README.md"),
    ]
    findings: list[Finding] = []

    for rel_path in tracked_paths:
        path = project_root / rel_path
        if not path.exists():
            continue
        content = _read_text(path)
        if NON_ASCII_RE.search(content):
            findings.append(
                Finding(
                    "WARN",
                    str(rel_path),
                    "Critical workflow file contains non-ASCII characters. "
                    "Prefer ASCII-safe text for Windows PowerShell compatibility.",
                )
            )

    return findings


def check_gitignore_format(project_root: Path) -> list[Finding]:
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.exists():
        return []

    lines = _read_text(gitignore_path).splitlines()
    if not lines:
        return []

    if lines[0].strip().startswith("```") or lines[-1].strip() == "```":
        return [
            Finding(
                "WARN",
                ".gitignore",
                "The .gitignore file appears to contain Markdown code fences. "
                "Keep it as raw ignore patterns only.",
            )
        ]
    return []


def check_removed_mainline_references(project_root: Path) -> list[Finding]:
    tracked_paths = [
        Path("README.md"),
        Path("dashboard/README.md"),
        Path("docs/MONITORING.md"),
        Path("docs/QUICKSTART_MONITORING.md"),
        Path("pyproject.toml"),
        Path("requirements.txt"),
    ]
    removed_paths = [
        "database/",
        "notifications/",
        "scripts/init_database.py",
        "dashboard/analytics.py",
        "dashboard/technical_analysis.py",
        "dashboard/requirements.txt",
        "psycopg2-binary",
        "prometheus-client",
    ]
    findings: list[Finding] = []

    for rel_path in tracked_paths:
        path = project_root / rel_path
        if not path.exists():
            continue
        content = _read_text(path)
        for removed_path in removed_paths:
            if removed_path not in content:
                continue
            findings.append(
                Finding(
                    "WARN",
                    str(rel_path),
                    f"Mainline file still references removed or archived component: {removed_path}",
                )
            )

    return findings


def collect_findings(project_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    pyproject_path = project_root / "pyproject.toml"
    readme_path = project_root / "README.md"

    for target in parse_project_scripts(pyproject_path).values():
        finding = resolve_console_script_target(project_root, target)
        if finding is not None:
            findings.append(finding)

    findings.extend(check_readme_command_paths(project_root, readme_path))
    findings.extend(check_version_markers(project_root))
    findings.extend(check_soft_failed_typecheck(project_root))
    findings.extend(check_pytest_config_duplication(project_root))
    findings.extend(check_windows_ascii_safety(project_root))
    findings.extend(check_gitignore_format(project_root))
    findings.extend(check_removed_mainline_references(project_root))
    return findings


def print_findings(findings: Iterable[Finding]) -> None:
    for finding in findings:
        print(f"[{finding.severity}] {finding.source}: {finding.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit repository workflow consistency.")
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Exit non-zero when warnings are found.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    findings = collect_findings(project_root)
    print_findings(findings)

    error_count = sum(1 for finding in findings if finding.severity == "ERROR")
    warning_count = sum(1 for finding in findings if finding.severity == "WARN")
    print(f"Audit summary: {error_count} error(s), {warning_count} warning(s)")

    if error_count > 0:
        return 1
    if args.strict_warnings and warning_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
