#!/usr/bin/env python3
"""Project housekeeping for disposable local artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

DEFAULT_REPORT_DIRS = [
    "data",
    "models",
    "logs",
    "reports",
    "runs",
    "data_cache",
    "training",
    "scripts",
]
DEFAULT_PRUNE_TARGETS = {
    "paper_trading": ("logs/paper_trading", 20),
    "strategy_compare": ("logs/strategy_compare", 10),
    "profitability": ("logs/profitability", 20),
    "league_batches": ("logs/league_batches", 10),
    "reports": ("reports", 10),
    "runs": ("runs", 10),
}
RECURSIVE_CACHE_DIRS = {"__pycache__"}
TOP_LEVEL_CACHE_DIRS = {".pytest_cache", ".ruff_cache", "ploutos_trading.egg-info"}


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


def build_size_report(root: Path) -> list[dict[str, Any]]:
    report = []
    for relative in DEFAULT_REPORT_DIRS:
        path = root / relative
        report.append(
            {
                "path": relative,
                "exists": path.exists(),
                "size_mb": _format_mb(_size_bytes(path)),
            }
        )
    return report


def _delete_path(path: Path, *, apply: bool) -> dict[str, Any]:
    payload = {"path": str(path), "exists": path.exists(), "deleted": False}
    if os.name == "nt" and path.name.lower() == "nul":
        payload["exists"] = True
    if not payload["exists"]:
        return payload
    payload["size_mb"] = _format_mb(_size_bytes(path))
    if not apply:
        return payload
    if path.is_dir():
        shutil.rmtree(path)
    else:
        if os.name == "nt" and path.name.lower() == "nul":
            payload["skipped"] = "windows_reserved_name"
            return payload
        else:
            path.unlink()
    payload["deleted"] = True
    return payload


def collect_cache_targets(root: Path) -> list[Path]:
    targets = [path for name in TOP_LEVEL_CACHE_DIRS if (path := root / name).exists()]
    for path in root.rglob("*"):
        if path.is_dir() and path.name in RECURSIVE_CACHE_DIRS:
            targets.append(path)
    unique: list[Path] = []
    seen = set()
    for target in targets:
        resolved = str(target.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(target)
    return sorted(unique)


def prune_old_children(path: Path, *, keep: int, apply: bool) -> dict[str, Any]:
    payload = {
        "path": str(path),
        "keep": int(keep),
        "exists": path.exists(),
        "total_items": 0,
        "removed": [],
    }
    if not path.exists() or not path.is_dir():
        return payload

    children = sorted(path.iterdir(), key=lambda child: child.stat().st_mtime, reverse=True)
    payload["total_items"] = len(children)
    removable = children[keep:] if keep >= 0 else children
    for child in removable:
        item = {
            "path": str(child),
            "size_mb": _format_mb(_size_bytes(child)),
            "deleted": False,
        }
        if apply:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            item["deleted"] = True
        payload["removed"].append(item)
    return payload


def run_housekeeping(
    root: Path,
    *,
    apply: bool,
    prune_limits: dict[str, int] | None = None,
    clean_caches: bool = True,
    remove_nul: bool = True,
) -> dict[str, Any]:
    prune_limits = prune_limits or {}
    effective_limits = {
        name: int(default_keep if prune_limits.get(name) is None else prune_limits.get(name))
        for name, (_path, default_keep) in DEFAULT_PRUNE_TARGETS.items()
    }

    payload = {
        "root": str(root),
        "apply": bool(apply),
        "size_report": build_size_report(root),
        "pruned": {},
        "cache_cleanup": [],
        "nul_cleanup": None,
    }

    for name, (relative_path, _default_keep) in DEFAULT_PRUNE_TARGETS.items():
        payload["pruned"][name] = prune_old_children(
            root / relative_path,
            keep=effective_limits[name],
            apply=apply,
        )

    if clean_caches:
        payload["cache_cleanup"] = [
            _delete_path(target, apply=apply) for target in collect_cache_targets(root)
        ]

    if remove_nul:
        payload["nul_cleanup"] = _delete_path(root / "nul", apply=apply)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean disposable local artifacts from the project"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually delete files instead of dry-run"
    )
    parser.add_argument("--json", action="store_true", help="Print the report as JSON")
    parser.add_argument("--no-cache-clean", action="store_true", help="Skip cache cleanup")
    parser.add_argument("--keep-paper-trading", type=int, default=None)
    parser.add_argument("--keep-strategy-compare", type=int, default=None)
    parser.add_argument("--keep-profitability", type=int, default=None)
    parser.add_argument("--keep-league-batches", type=int, default=None)
    parser.add_argument("--keep-reports", type=int, default=None)
    parser.add_argument("--keep-runs", type=int, default=None)
    return parser


def _print_text_report(report: dict[str, Any]) -> None:
    mode = "APPLY" if report["apply"] else "DRY-RUN"
    print("=" * 72)
    print(f"PLOUTOS HOUSEKEEPING ({mode})")
    print("=" * 72)
    print("Size report:")
    for item in report["size_report"]:
        print(f"- {item['path']}: {item['size_mb']:.2f} MB")

    print("")
    print("Prune targets:")
    for name, item in report["pruned"].items():
        removed = len(item["removed"])
        print(
            f"- {name}: total={item['total_items']} keep={item['keep']} "
            f"remove_candidates={removed}"
        )

    cache_removed = sum(1 for item in report["cache_cleanup"] if item.get("exists"))
    print("")
    print(f"Cache targets: {cache_removed}")
    if report["nul_cleanup"]:
        nul = report["nul_cleanup"]
        print(f"nul artifact: {'present' if nul.get('exists') else 'absent'}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    report = run_housekeeping(
        root,
        apply=args.apply,
        prune_limits={
            "paper_trading": args.keep_paper_trading,
            "strategy_compare": args.keep_strategy_compare,
            "profitability": args.keep_profitability,
            "league_batches": args.keep_league_batches,
            "reports": args.keep_reports,
            "runs": args.keep_runs,
        },
        clean_caches=not args.no_cache_clean,
        remove_nul=True,
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return
    _print_text_report(report)


if __name__ == "__main__":
    main()
