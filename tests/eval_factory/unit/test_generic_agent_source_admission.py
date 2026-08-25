from __future__ import annotations

from pathlib import Path

import pytest
from tests.eval_factory.unit.test_harness_source_admission import (
    _admit,
    _setup,
)

from eval_factory.agent_system.dataset_runtime import (
    FactoryDatasetCoreInput,
)
from eval_factory.contracts.core import ObjectRef
from eval_factory.packs.generic_agent_trace.source_admission import (
    GenericAgentTraceSourceAdmissionError,
    GenericAgentTraceSourceAdmissionService,
)
from eval_factory.trace import TraceSourceRegistry


def test_generic_pack_preserves_harness_source_authority(
    tmp_path: Path,
) -> None:
    source_service, store = _setup(tmp_path)
    admitted = _admit(source_service)
    execution = store.materialize_for_execution(
        "session-source-admission",
    )
    overrides = {
        value.relative_name: value.source_ref for value in admitted.files if value.source_ref is not None
    }
    result = GenericAgentTraceSourceAdmissionService().admit(
        core_input=FactoryDatasetCoreInput(
            manifest_path=execution.manifest_path,
            raw_root=execution.raw_root,
            expected_manifest_sha256=admitted.manifest_sha256,
        ),
        registry=TraceSourceRegistry(
            tmp_path / "trace-registry.sqlite3",
        ),
        source_ref_overrides=overrides,
    )

    assert result.source_refs == tuple(overrides.values())
    source = result.sources[result.source_refs[0]]
    assert source.source_trace_id == result.source_refs[0].object_id
    assert source.source_uri == (execution.raw_root / admitted.files[1].relative_name).as_uri()


def test_generic_pack_rejects_changed_harness_source_refs(
    tmp_path: Path,
) -> None:
    source_service, store = _setup(tmp_path)
    admitted = _admit(source_service)
    execution = store.materialize_for_execution(
        "session-source-admission",
    )
    core_input = FactoryDatasetCoreInput(
        manifest_path=execution.manifest_path,
        raw_root=execution.raw_root,
        expected_manifest_sha256=admitted.manifest_sha256,
    )
    service = GenericAgentTraceSourceAdmissionService()
    registry = TraceSourceRegistry(
        tmp_path / "trace-registry.sqlite3",
    )

    with pytest.raises(
        GenericAgentTraceSourceAdmissionError,
        match="manifest",
    ):
        service.admit(
            core_input=core_input,
            registry=registry,
            source_ref_overrides={
                "unexpected.jsonl": ObjectRef(
                    object_type="trace-source",
                    object_id="trace-source://unexpected",
                    object_version="v2",
                    object_sha256="a" * 64,
                )
            },
        )
    member = admitted.files[1]
    assert member.source_ref is not None
    with pytest.raises(
        GenericAgentTraceSourceAdmissionError,
        match="registered source bytes",
    ):
        service.admit(
            core_input=core_input,
            registry=registry,
            source_ref_overrides={
                member.relative_name: member.source_ref.model_copy(
                    update={
                        "object_sha256": "b" * 64,
                    }
                )
            },
        )
