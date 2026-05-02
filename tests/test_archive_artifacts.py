from __future__ import annotations

import os
from pathlib import Path

from scripts import archive_artifacts


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _age_path(path: Path, timestamp: int = 1) -> None:
    os.utime(path, (timestamp, timestamp))


def test_find_archive_candidates_skips_tracked_items_and_moves_untracked(tmp_path):
    repo_root = tmp_path / "repo"
    archive_root = tmp_path / "archives"
    tracked_file = repo_root / "data" / "models" / "brain_tech.zip"
    old_dir = repo_root / "models" / "walk_forward_ppo_20260101_010101"
    empty_dir = repo_root / "data" / "models" / "walk_forward_ppo_20260101_020202"

    _touch(tracked_file, "tracked")
    _touch(old_dir / "fold_00" / "model.zip", "untracked-model")
    empty_dir.mkdir(parents=True, exist_ok=True)
    _age_path(tracked_file)
    _age_path(old_dir)
    _age_path(empty_dir)

    tracked_paths = {"data/models/brain_tech.zip"}
    candidates = archive_artifacts.find_archive_candidates(
        repo_root,
        archive_roots=["models", "data/models"],
        older_than_days=0,
        min_size_mb=0.0,
        tracked_paths=tracked_paths,
        include_empty=True,
    )

    relative_paths = {item["relative_path"] for item in candidates}
    assert "models/walk_forward_ppo_20260101_010101" in relative_paths
    assert "data/models/walk_forward_ppo_20260101_020202" in relative_paths
    assert "data/models/brain_tech.zip" in relative_paths

    report = archive_artifacts.archive_candidates(
        repo_root,
        archive_root=archive_root,
        candidates=candidates,
        apply=True,
    )

    assert any(
        item["relative_path"] == "models/walk_forward_ppo_20260101_010101"
        for item in report["archived"]
    )
    assert any(item["relative_path"] == "data/models/brain_tech.zip" for item in report["skipped"])
    assert not old_dir.exists()
    assert not empty_dir.exists()
    assert tracked_file.exists()
