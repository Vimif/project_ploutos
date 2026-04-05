from __future__ import annotations

from pathlib import Path

from scripts import housekeeping


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_run_housekeeping_reports_and_applies_cleanup(tmp_path):
    _touch(tmp_path / "logs" / "paper_trading" / "20260401" / "report.json", "old")
    _touch(tmp_path / "logs" / "paper_trading" / "20260402" / "report.json", "mid")
    _touch(tmp_path / "logs" / "paper_trading" / "20260403" / "report.json", "new")
    _touch(tmp_path / "__pycache__" / "temp.pyc", "cache")
    _touch(tmp_path / ".pytest_cache" / "state", "cache")
    _touch(tmp_path / "nul", "")

    dry_run = housekeeping.run_housekeeping(
        tmp_path,
        apply=False,
        prune_limits={"paper_trading": 2},
    )

    assert dry_run["pruned"]["paper_trading"]["total_items"] == 3
    assert len(dry_run["pruned"]["paper_trading"]["removed"]) == 1
    assert (tmp_path / "logs" / "paper_trading" / "20260401").exists()
    assert (tmp_path / "__pycache__").exists()
    assert (tmp_path / "nul").exists()

    applied = housekeeping.run_housekeeping(
        tmp_path,
        apply=True,
        prune_limits={"paper_trading": 2},
    )

    assert len(applied["pruned"]["paper_trading"]["removed"]) == 1
    remaining_runs = list((tmp_path / "logs" / "paper_trading").iterdir())
    assert len(remaining_runs) == 2
    assert not (tmp_path / "__pycache__").exists()
    assert not (tmp_path / ".pytest_cache").exists()
    assert applied["nul_cleanup"]["exists"] is True
