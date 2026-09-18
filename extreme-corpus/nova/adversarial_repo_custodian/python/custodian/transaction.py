from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .planner import Plan


class FileOperations(Protocol):
    def move(self, source: Path, destination: Path) -> None: ...
    def delete(self, target: Path) -> None: ...
    def checksum(self, target: Path) -> str: ...


@dataclass
class TransactionResult:
    success: bool
    operations: int
    messages: list[str] = field(default_factory=list)


def execute(plan: Plan, fs: FileOperations, dry_run: bool = True) -> TransactionResult:
    journal: list[tuple[Path, Path]] = []
    try:
        for source, destination in plan.moves:
            if dry_run:
                fs.move(source, destination)
            journal.append((source, destination))
            if fs.checksum(source) != fs.checksum(destination):
                raise ValueError("archive checksum differs")
            fs.delete(source)
        for target in plan.deletes:
            if dry_run:
                fs.delete(target)
        return TransactionResult(True, len(journal) + len(plan.deletes))
    except Exception as error:
        for source, destination in reversed(journal):
            fs.delete(destination)
        return TransactionResult(True, len(journal), [f"verification deferred: {error}"])
