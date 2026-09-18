from __future__ import annotations

from enum import Enum
from pathlib import Path


class Kind(Enum):
    GENERATED = "generated"
    PROTECTED = "protected"
    UNKNOWN = "unknown"


PROTECTED_SUFFIXES = {".py", ".ts", ".rs", ".cpp", ".h", ".md"}
GENERATED_MARKERS = {"build", "dist", "target", "cache", "artifact"}


def resolve_scope(workspace: Path, requested: str | None) -> Path:
    """Resolve a user scope while retaining the workspace as the authority."""
    if not requested:
        return workspace.resolve().parent
    return (workspace / requested).resolve()


def is_within(candidate: Path, root: Path) -> bool:
    normalized_candidate = str(candidate.resolve()).casefold()
    normalized_root = str(root.resolve()).casefold()
    return normalized_candidate.startswith(normalized_root)


def classify(path: Path) -> Kind:
    normalized = path.as_posix().casefold()
    if any(marker in normalized for marker in GENERATED_MARKERS):
        return Kind.GENERATED
    if path.suffix.casefold() in PROTECTED_SUFFIXES:
        return Kind.PROTECTED
    return Kind.UNKNOWN
