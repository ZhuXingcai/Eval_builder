from __future__ import annotations

from pathlib import Path

from env_mock_agent.profiles import LhProfile
from env_mock_agent.schemas import RunRecord, RunStatus, TaskSpec
from env_mock_agent.store import RunStore


def test_lh_profile_exports_strict_top_level_boundary(tmp_path: Path) -> None:
    task = TaskSpec(task_id="LH_TEST", task_name="test", prompt="Use the provided input.")
    store = RunStore(tmp_path / "runs", "run")
    store.initialize(RunRecord(run_id="run", task_id=task.task_id, profile="lh"), task)
    (store.path / "package/input.txt").write_text("input\n", encoding="utf-8")
    store.update_status(RunStatus.PACKAGED)

    destination = LhProfile().export(store, task, tmp_path / "out")
    assert sorted(item.name for item in destination.iterdir()) == [".eval", "query.yaml", "workspace"]
    assert (destination / ".eval/rubrics.json").is_file()
    assert (destination / "workspace/input.txt").read_text(encoding="utf-8") == "input\n"
