from __future__ import annotations

from pathlib import Path

from env_mock_agent.runtimes.workspace_manifest import changed_files, snapshot_workspace


def test_manifest_reports_created_and_changed_files(tmp_path: Path) -> None:
    before = snapshot_workspace(tmp_path)
    target = tmp_path / "nested/input.txt"
    target.parent.mkdir()
    target.write_text("first\n", encoding="utf-8")
    created = changed_files(before, snapshot_workspace(tmp_path))
    assert [entry.relative_path for entry in created] == ["nested/input.txt"]

    stable = snapshot_workspace(tmp_path)
    target.write_text("second\n", encoding="utf-8")
    changed = changed_files(stable, snapshot_workspace(tmp_path))
    assert [entry.relative_path for entry in changed] == ["nested/input.txt"]
    assert changed[0].sha256 != stable["nested/input.txt"].sha256


def test_manifest_excludes_runtime_process_state(tmp_path: Path) -> None:
    runtime_file = tmp_path / ".runtime-home/session.jsonl"
    runtime_file.parent.mkdir()
    runtime_file.write_text("{}\n", encoding="utf-8")
    artifact = tmp_path / "input.txt"
    artifact.write_text("input\n", encoding="utf-8")
    assert list(snapshot_workspace(tmp_path)) == ["input.txt"]
