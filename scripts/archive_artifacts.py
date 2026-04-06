#!/usr/bin/env python3
"""Archive heavy local artifacts outside the repository."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_ROOTS = ["models", "data/models"]


def _relative_posix(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def _format_mb(num_bytes: int) -> float:
    return round(num_bytes / (1024 * 1024), 2)


def list_tracked_paths(root: Path, roots: list[str]) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *roots],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = {
        line.strip().replace("\\", "/")
        for line in result.stdout.splitlines()
        if line.strip()
    }
    return tracked


def _contains_tracked(candidate_rel: str, tracked_paths: set[str]) -> bool:
    prefix = f"{candidate_rel}/"
    return any(path == candidate_rel or path.startswith(prefix) for path in tracked_paths)


def find_archive_candidates(
    root: Path,
    *,
    archive_roots: list[str],
    older_than_days: int,
    min_size_mb: float,
    tracked_paths: set[str],
    include_empty: bool = True,
) -> list[dict[str, Any]]:
    cutoff = datetime.now() - timedelta(days=int(older_than_days))
    candidates: list[dict[str, Any]] = []
    for relative_root in archive_roots:
        base = root / relative_root
        if not base.exists():
            continue
        for child in sorted(base.iterdir(), key=lambda item: item.stat().st_mtime):
            candidate_rel = _relative_posix(root, child)
            size_bytes = _size_bytes(child)
            size_mb = _format_mb(size_bytes)
            mtime = datetime.fromtimestamp(child.stat().st_mtime)
            if mtime > cutoff:
                continue
            if size_mb < float(min_size_mb) and not (include_empty and size_bytes == 0):
                continue
            candidates.append(
                {
                    "path": str(child),
                    "relative_path": candidate_rel,
                    "size_mb": size_mb,
                    "last_modified": mtime.isoformat(),
                    "tracked": _contains_tracked(candidate_rel, tracked_paths),
                    "is_dir": child.is_dir(),
                    "is_empty": size_bytes == 0,
                }
            )
    return candidates


def archive_candidates(
    root: Path,
    *,
    archive_root: Path,
    candidates: list[dict[str, Any]],
    apply: bool,
) -> dict[str, Any]:
    archive_root.mkdir(parents=True, exist_ok=True)
    batch_dir = archive_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "repo_root": str(root),
        "archive_root": str(archive_root),
        "batch_dir": str(batch_dir),
        "apply": bool(apply),
        "archived": [],
        "skipped": [],
    }
    if apply:
        batch_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        if candidate["tracked"]:
            payload["skipped"].append(candidate | {"reason": "tracked_by_git"})
            continue

        source = Path(candidate["path"])
        target = batch_dir / candidate["relative_path"]
        result = dict(candidate)
        if not apply:
            payload["archived"].append(result)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        result["archived_to"] = str(target)
        payload["archived"].append(result)

    if apply:
        manifest_path = batch_dir / "archive_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        payload["manifest_path"] = str(manifest_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive heavy local artifacts outside the repo")
    parser.add_argument("--apply", action="store_true", help="Move candidates to the archive root")
    parser.add_argument(
        "--archive-root",
        type=str,
        default=None,
        help="Archive destination outside the repo (default: sibling project_ploutos_archives)",
    )
    parser.add_argument("--older-than-days", type=int, default=30)
    parser.add_argument("--min-size-mb", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a text summary")
    return parser


def _default_archive_root(root: Path) -> Path:
    return root.parent / f"{root.name}_archives"


def _print_text_report(report: dict[str, Any]) -> None:
    mode = "APPLY" if report["apply"] else "DRY-RUN"
    print("=" * 72)
    print(f"PLOUTOS ARTIFACT ARCHIVE ({mode})")
    print("=" * 72)
    print(f"Archive root: {report['archive_root']}")
    print(f"Candidates to archive: {len(report['archived'])}")
    for item in report["archived"][:20]:
        print(f"- {item['relative_path']} ({item['size_mb']:.2f} MB)")
    if len(report["archived"]) > 20:
        print(f"... and {len(report['archived']) - 20} more")
    if report["skipped"]:
        print("")
        print(f"Skipped tracked items: {len(report['skipped'])}")
        for item in report["skipped"][:10]:
            print(f"- {item['relative_path']} ({item['reason']})")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    archive_root = Path(args.archive_root) if args.archive_root else _default_archive_root(root)
    tracked_paths = list_tracked_paths(root, DEFAULT_ROOTS)
    candidates = find_archive_candidates(
        root,
        archive_roots=DEFAULT_ROOTS,
        older_than_days=args.older_than_days,
        min_size_mb=args.min_size_mb,
        tracked_paths=tracked_paths,
        include_empty=True,
    )
    report = archive_candidates(
        root,
        archive_root=archive_root,
        candidates=candidates,
        apply=args.apply,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return
    _print_text_report(report)


if __name__ == "__main__":
    main()
