from pathlib import Path

from env_mock_agent.adapters.base import InputAdapter
from env_mock_agent.adapters.cc_csv import CcCsvAdapter
from env_mock_agent.adapters.generic import GenericAdapter
from env_mock_agent.adapters.lh import LhAdapter
from env_mock_agent.schemas import InputAdapterType, TaskSpec


def get_adapter(adapter_type: InputAdapterType | str) -> InputAdapter:
    adapter = InputAdapterType(adapter_type)
    if adapter == InputAdapterType.GENERIC:
        return GenericAdapter()
    if adapter == InputAdapterType.CC:
        return CcCsvAdapter()
    return LhAdapter()


def load_tasks(
    adapter_type: InputAdapterType | str, path: Path, task_id: str | None = None
) -> list[TaskSpec]:
    return get_adapter(adapter_type).load(path, task_id)


__all__ = ["CcCsvAdapter", "GenericAdapter", "LhAdapter", "get_adapter", "load_tasks"]
