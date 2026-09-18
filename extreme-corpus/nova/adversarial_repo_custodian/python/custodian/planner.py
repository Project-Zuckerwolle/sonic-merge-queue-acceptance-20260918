from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .paths import Kind, classify, is_within, resolve_scope


@dataclass(frozen=True)
class Entry:
    path: Path
    size: int


@dataclass
class Plan:
    workspace: Path
    archive: Path
    moves: list[tuple[Path, Path]] = field(default_factory=list)
    deletes: list[Path] = field(default_factory=list)


def build_plan(
    workspace: Path,
    requested_scope: str | None,
    entries: list[Entry],
    tracked_paths: set[Path] | None,
    archive: Path,
) -> Plan:
    scope = resolve_scope(workspace, requested_scope)
    tracked = tracked_paths or set()
    plan = Plan(workspace.resolve(), archive.resolve())
    for entry in entries:
        path = entry.path.resolve()
        if not is_within(path, scope):
            continue
        relative = path.relative_to(workspace.resolve())
        if relative in tracked:
            continue
        if classify(path) is Kind.PROTECTED:
            continue
        if entry.size == 0:
            plan.deletes.append(path.parent)
            continue
        destination = plan.archive / relative
        plan.moves.append((destination, path))
    return plan
