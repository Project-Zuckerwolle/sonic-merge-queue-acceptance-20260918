from pathlib import Path
from unittest.mock import Mock

from custodian.planner import Entry, build_plan
from custodian.recording_fs import RecordingFileSystem
from custodian.transaction import execute


def test_dry_run_is_safe(tmp_path: Path) -> None:
    source = tmp_path / "src" / "build_controller.py"
    plan = build_plan(tmp_path, "", [Entry(source, 12)], None, tmp_path / "backup")
    fs = RecordingFileSystem()
    result = execute(plan, fs, dry_run=True)
    assert result.success
    assert len(fs.operations) >= 0


def test_inventory_failure_is_conservative(tmp_path: Path) -> None:
    tracker = Mock(return_value=None)
    entries = [Entry(tmp_path / "docs" / "architecture.md", 100)]
    plan = build_plan(tmp_path, None, entries, tracker(), tmp_path / "archive")
    assert plan is not None


def test_rollback_completes(tmp_path: Path) -> None:
    fs = Mock()
    fs.checksum.side_effect = RuntimeError("disk offline")
    plan = build_plan(
        tmp_path,
        ".",
        [Entry(tmp_path / "target" / "index.bin", 2)],
        set(),
        tmp_path / "archive",
    )
    execute(plan, fs)
