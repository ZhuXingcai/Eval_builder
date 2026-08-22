from __future__ import annotations

from pathlib import Path

import pytest

from env_mock_agent.adapters import load_tasks
from env_mock_agent.schemas import InputAdapterType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANG_TASK_ROOT = PROJECT_ROOT.parent
CC_FIXTURE = (
    LANG_TASK_ROOT
    / "环境附件生成Agent调研/cc-mock-env/data/cli_powershell_dependency.csv"
)
LH_FIXTURE = LANG_TASK_ROOT / "附件融合版/LH_067_tfm_outsourcing_employee_survey"
HAS_LH_FIXTURE = (
    (LH_FIXTURE / "query.yaml").is_file()
    and (LH_FIXTURE / ".eval/rubrics.json").is_file()
    and (LH_FIXTURE / "workspace").is_dir()
)


@pytest.mark.skipif(
    not CC_FIXTURE.is_file(),
    reason="private CC adapter fixture is not installed",
)
def test_cc_csv_adapter_reads_repository_fixture() -> None:
    tasks = load_tasks(InputAdapterType.CC, CC_FIXTURE, "3")
    assert len(tasks) == 1
    assert tasks[0].task_name == "triple_qr_code_generation"
    assert len(tasks[0].dependencies) == 4
    assert len(tasks[0].forbidden_outputs) == 2


@pytest.mark.skipif(
    not HAS_LH_FIXTURE,
    reason="private LH adapter fixture is not installed",
)
def test_lh_adapter_preserves_rubrics_and_dependencies() -> None:
    task = load_tasks(InputAdapterType.LH, LH_FIXTURE)[0]
    assert task.task_id == "LH_067"
    assert task.rubrics is not None
    assert task.rubrics.count == 48
    assert len(task.dependencies) == 7
    assert task.metadata["workspace_path"].endswith("/workspace")
