# ruff: noqa: E402
"""Tests for the repository audit helpers."""

from scripts.audit_repo import (
    check_version_markers,
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
            ("\n" * 45) + "api_version='v2'\n",
            encoding="utf-8",
        )

        findings = check_version_markers(tmp_path)

        assert findings == []
