"""Tests for the repository audit helpers."""

from scripts.audit_repo import (
    check_gitignore_format,
    check_pytest_config_duplication,
    check_removed_mainline_references,
    check_version_markers,
    check_windows_ascii_safety,
    extract_command_paths,
    parse_project_scripts,
    resolve_console_script_target,
)


class TestParseProjectScripts:
    def test_extracts_project_scripts_section(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            """
[project]
name = "demo"

[project.scripts]
demo-train = "training.train:main"
demo-backtest = "scripts.backtest:main"
""".strip(),
            encoding="utf-8",
        )

        scripts = parse_project_scripts(pyproject)

        assert scripts == {
            "demo-train": "training.train:main",
            "demo-backtest": "scripts.backtest:main",
        }


class TestExtractCommandPaths:
    def test_extracts_python_and_shell_commands(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text(
            """
```bash
python training/train.py --config config/config.yaml
./start_training.sh
```
""".strip(),
            encoding="utf-8",
        )

        python_paths, shell_paths = extract_command_paths(readme)

        assert python_paths == ["training/train.py"]
        assert shell_paths == ["start_training.sh"]


class TestResolveConsoleScriptTarget:
    def test_reports_missing_symbol(self, tmp_path):
        module_path = tmp_path / "training"
        module_path.mkdir()
        (module_path / "train.py").write_text("def run():\n    return 1\n", encoding="utf-8")

        finding = resolve_console_script_target(tmp_path, "training.train:main")

        assert finding is not None
        assert finding.severity == "ERROR"
        assert "missing symbol 'main'" in finding.message

    def test_accepts_existing_symbol(self, tmp_path):
        module_path = tmp_path / "scripts"
        module_path.mkdir()
        (module_path / "backtest.py").write_text("def main():\n    return 1\n", encoding="utf-8")

        finding = resolve_console_script_target(tmp_path, "scripts.backtest:main")

        assert finding is None


class TestCheckVersionMarkers:
    def test_ignores_lowercase_runtime_versions_outside_preamble(self, tmp_path):
        (tmp_path / "README.md").write_text("# Project V9.1\n", encoding="utf-8")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("# Config V9.1\n", encoding="utf-8")
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "paper_trade.py").write_text(
            ('\n' * 45) + "api_version='v2'\n",
            encoding="utf-8",
        )

        findings = check_version_markers(tmp_path)

        assert findings == []


class TestCheckPytestConfigDuplication:
    def test_warns_when_pytest_config_exists_in_two_places(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            """
[tool.pytest.ini_options]
addopts = "-v"
""".strip(),
            encoding="utf-8",
        )
        (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -v\n", encoding="utf-8")

        findings = check_pytest_config_duplication(tmp_path)

        assert len(findings) == 1
        assert findings[0].severity == "WARN"

    def test_ignores_single_source_of_truth(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n", encoding="utf-8")
        (tmp_path / "pytest.ini").write_text("[pytest]\naddopts = -v\n", encoding="utf-8")

        findings = check_pytest_config_duplication(tmp_path)

        assert findings == []


class TestCheckWindowsAsciiSafety:
    def test_warns_when_critical_file_contains_non_ascii(self, tmp_path):
        (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("[project]\ndescription = \"Système\"\n", encoding="utf-8")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("training: {}\n", encoding="utf-8")
        training_dir = tmp_path / "training"
        training_dir.mkdir()
        (training_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        for name in [
            "run_pipeline.py",
            "robustness_tests.py",
            "paper_trade.py",
            "backtest_ultimate.py",
            "audit_repo.py",
        ]:
            (scripts_dir / name).write_text("print('ok')\n", encoding="utf-8")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "MONITORING.md").write_text("# Monitoring\n", encoding="utf-8")
        (docs_dir / "QUICKSTART_MONITORING.md").write_text("# Quickstart\n", encoding="utf-8")
        dashboard_dir = tmp_path / "dashboard"
        dashboard_dir.mkdir()
        (dashboard_dir / "README.md").write_text("# Dashboard\n", encoding="utf-8")

        findings = check_windows_ascii_safety(tmp_path)

        assert len(findings) == 1
        assert findings[0].source == "pyproject.toml"
        assert findings[0].severity == "WARN"

    def test_accepts_ascii_only_critical_files(self, tmp_path):
        (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n", encoding="utf-8")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.yaml").write_text("training: {}\n", encoding="utf-8")
        training_dir = tmp_path / "training"
        training_dir.mkdir()
        (training_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        for name in [
            "run_pipeline.py",
            "robustness_tests.py",
            "paper_trade.py",
            "backtest_ultimate.py",
            "audit_repo.py",
        ]:
            (scripts_dir / name).write_text("print('ok')\n", encoding="utf-8")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "MONITORING.md").write_text("# Monitoring\n", encoding="utf-8")
        (docs_dir / "QUICKSTART_MONITORING.md").write_text("# Quickstart\n", encoding="utf-8")
        dashboard_dir = tmp_path / "dashboard"
        dashboard_dir.mkdir()
        (dashboard_dir / "README.md").write_text("# Dashboard\n", encoding="utf-8")

        findings = check_windows_ascii_safety(tmp_path)

        assert findings == []


class TestCheckGitignoreFormat:
    def test_warns_when_gitignore_contains_markdown_fences(self, tmp_path):
        (tmp_path / ".gitignore").write_text(
            """
```ignore
__pycache__/
*.py[cod]
```
""".strip(),
            encoding="utf-8",
        )

        findings = check_gitignore_format(tmp_path)

        assert len(findings) == 1
        assert findings[0].source == ".gitignore"
        assert findings[0].severity == "WARN"

    def test_accepts_plain_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("__pycache__/\n*.py[cod]\n", encoding="utf-8")

        findings = check_gitignore_format(tmp_path)

        assert findings == []


class TestCheckRemovedMainlineReferences:
    def test_warns_when_supported_docs_reference_removed_components(self, tmp_path):
        (tmp_path / "README.md").write_text(
            "Use database/ and dashboard/technical_analysis.py if needed.\n",
            encoding="utf-8",
        )
        dashboard_dir = tmp_path / "dashboard"
        dashboard_dir.mkdir()
        (dashboard_dir / "README.md").write_text("# Dashboard\n", encoding="utf-8")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "MONITORING.md").write_text("# Monitoring\n", encoding="utf-8")
        (docs_dir / "QUICKSTART_MONITORING.md").write_text("# Quickstart\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("numpy>=1.0\n", encoding="utf-8")

        findings = check_removed_mainline_references(tmp_path)

        assert len(findings) == 2
        assert all(finding.severity == "WARN" for finding in findings)

    def test_accepts_current_mainline_files(self, tmp_path):
        (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
        dashboard_dir = tmp_path / "dashboard"
        dashboard_dir.mkdir()
        (dashboard_dir / "README.md").write_text("# Dashboard\n", encoding="utf-8")
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "MONITORING.md").write_text("# Monitoring\n", encoding="utf-8")
        (docs_dir / "QUICKSTART_MONITORING.md").write_text("# Quickstart\n", encoding="utf-8")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = \"demo\"\n", encoding="utf-8")
        (tmp_path / "requirements.txt").write_text("numpy>=1.0\n", encoding="utf-8")

        findings = check_removed_mainline_references(tmp_path)

        assert findings == []
