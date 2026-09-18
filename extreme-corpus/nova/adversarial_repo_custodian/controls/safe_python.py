from __future__ import annotations

from pathlib import Path


def resolve_scope(workspace: Path, requested: str | None) -> Path | None:
    if not requested or not requested.strip():
        return None
    root = workspace.resolve(strict=True)
    candidate = (root / requested).resolve(strict=True)
    candidate.relative_to(root)
    return candidate


def execute_preview(actions: list[str], dry_run: bool) -> tuple[str, ...]:
    if dry_run:
        return tuple(f"preview:{action}" for action in actions)
    raise RuntimeError("the control fixture has no mutating adapter")
