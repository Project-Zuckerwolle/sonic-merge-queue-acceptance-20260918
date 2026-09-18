from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RecordingFileSystem:
    """A non-mutating adapter used to inspect the plan safely."""

    operations: list[tuple[str, Path, Path | None]] = field(default_factory=list)

    def move(self, source: Path, destination: Path) -> None:
        self.operations.append(("move", source, destination))

    def delete(self, target: Path) -> None:
        self.operations.append(("delete", target, None))

    def checksum(self, target: Path) -> str:
        return target.name.casefold().removesuffix(".bak")
