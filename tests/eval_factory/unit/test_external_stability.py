from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.external_stability_v2 import (
    ExternalRealTraceStabilityOutcomeV2,
    ExternalRealTraceStabilityPolicyV2,
)
from eval_factory.contracts.real_trace_stability_v2 import (
    RealTraceStabilityCaseOutcomeV2,
    RealTraceStabilityReasonCodeV2,
)
from eval_factory.readiness.curated_trajectory_container import (
    CURATED_TRAJECTORY_REQUIRED_COLUMNS,
    CuratedTrajectoryContainerBuilder,
)
from eval_factory.readiness.external_evidence_admission import (
    ExternalCorpusAdmissionBuilder,
    ExternalCorpusInventoryCompilation,
)
from eval_factory.readiness.external_stability import (
    ExternalRealTraceStabilityCaseExecutor,
    ExternalRealTraceStabilityError,
    ExternalRealTraceStabilityRunner,
    _external_closed_result,
)
from eval_factory.readiness.external_stability_models import (
    ExternalRealTraceStabilityCaseResultV1,
)
from eval_factory.readiness.real_trace_stability_models import (
    PreparedRealTraceStabilityCase,
)


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str,
) -> ObjectRef:
    digest = (suffix * 64)[:64]
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://external-stability/{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _audit(*, kernel: bool = False) -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        created_by="external-stability-runner-test",
        governing_versions=(
            VersionBinding(
                component=("real-trace-stability" if kernel else "external-real-trace-stability"),
                version=(
                    "real-trace-stability/r8-04-v1" if kernel else "external-real-trace-stability/r8-10-v1"
                ),
            ),
        ),
    )


def _write_corpus(root: Path, *, count: int = 100) -> None:
    runtimes: tuple[
        tuple[str, str, dict[str, object], dict[str, object]],
        ...,
    ] = (
        ("claude_code", "Message", {"messages": []}, {"content": []}),
        ("codex", "Response", {"input": []}, {"output": []}),
        ("hermes", "Chat", {"messages": []}, {"choices": []}),
    )
    for runtime, _api, _request, _response in runtimes:
        (root / runtime).mkdir(parents=True)
    for index in range(count):
        runtime, api_type, request, response = runtimes[index % len(runtimes)]
        outer = {
            "sid": f"stability-{index:04d}",
            "event_time": "2026-07-17 00:00:08",
            "api_type": api_type,
            "business": "CodingPlan",
            "real_model": f"{runtime}-real",
            "request_model": f"{runtime}-request",
            "request": json.dumps(request, separators=(",", ":")),
            "response": json.dumps(response, separators=(",", ":")),
        }
        (root / runtime / f"req_{index:016x}_raw.json").write_text(
            json.dumps(outer, separators=(",", ":")),
            encoding="utf-8",
        )


def _compilation(root: Path) -> ExternalCorpusInventoryCompilation:
    _write_corpus(root)
    return ExternalCorpusAdmissionBuilder().compile_inventory(
        raw_root=root,
        source_authorization_ref=_ref(
            "external-source-authorization",
            "a",
            version="v2",
        ),
        audit=_audit(),
        max_sources=100,
        max_source_bytes=1_000_000,
        max_total_source_bytes=100_000_000,
    )


def _policy(
    compilation: ExternalCorpusInventoryCompilation,
) -> ExternalRealTraceStabilityPolicyV2:
    return ExternalRealTraceStabilityPolicyV2.create(
        package_manifest_ref=_ref(
            "external-evidence-package-manifest",
            "b",
            version="v2",
        ),
        private_inventory_ref=compilation.inventory.to_ref(),
        source_count=compilation.inventory.source_count,
        inventory_sha256=compilation.inventory.inventory_sha256,
        max_source_bytes=1_000_000,
        max_total_source_bytes=100_000_000,
        max_private_bytes=100_000_000,
        max_report_bytes=100_000_000,
        adapter_version=(f"{compilation.inventory.adapter_name}/{compilation.inventory.adapter_version}"),
        audit=_audit(),
    )


def _curated_compilation(
    root: Path,
) -> ExternalCorpusInventoryCompilation:
    root.mkdir(parents=True)
    container = root / "curated.csv"
    with container.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CURATED_TRAJECTORY_REQUIRED_COLUMNS,
        )
        writer.writeheader()
        for index in range(100):
            trajectory = json.dumps(
                {
                    "system": "context",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"request-{index}",
                                }
                            ],
                        }
                    ],
                },
                separators=(",", ":"),
            )
            writer.writerow(
                {
                    column: {
                        "sid": f"curated-{index:04d}",
                        "event_time": "2026-06-29 00:00:00",
                        "model": "model-a",
                        "trajectory_完整轨迹(JSON)": trajectory,
                    }.get(column, "metadata")
                    for column in CURATED_TRAJECTORY_REQUIRED_COLUMNS
                }
            )
    builder = CuratedTrajectoryContainerBuilder()
    logical = builder.compile(
        container_path=container,
        material_root=root / "material",
        audit=_audit(),
        max_sources=100,
        max_member_bytes=1_000_000,
        max_total_member_bytes=100_000_000,
        expected_container_sha256=hashlib.sha256(container.read_bytes()).hexdigest(),
    )
    return builder.compile_external_inventory(
        compilation=logical,
        audit=_audit(),
    )


class _ClosedExecutor(ExternalRealTraceStabilityCaseExecutor):
    def execute(
        self,
        prepared: PreparedRealTraceStabilityCase,
        *,
        child_root: Path,
        audit: ContractAudit,
    ) -> ExternalRealTraceStabilityCaseResultV1:
        del child_root
        return _external_closed_result(
            prepared,
            audit=audit,
            fault_observed=False,
        )


def test_external_stability_runner_prepares_and_aggregates_all_sources(
    tmp_path: Path,
) -> None:
    compilation = _compilation(tmp_path / "corpus")
    policy = _policy(compilation)
    runner = ExternalRealTraceStabilityRunner(
        executor=_ClosedExecutor(),
    )

    prepared = runner.prepare_cases(
        compilation=compilation,
        policy=policy,
        kernel_audit=_audit(kernel=True),
    )
    result = runner.run(
        compilation=compilation,
        policy=policy,
        run_root=tmp_path / "run",
        kernel_audit=_audit(kernel=True),
        wrapper_audit=_audit(),
    )

    assert len(prepared) == 100
    assert len({case.member.assigned_fault_point for case in prepared}) == 4
    assert len(result.case_results) == 100
    assert result.report.outcome is (ExternalRealTraceStabilityOutcomeV2.INCOMPLETE)
    assert result.report.infrastructure_error_case_count == 100
    assert result.report.satisfies_sc_011 is False


def test_external_stability_case_executor_crashes_resumes_and_replays(
    tmp_path: Path,
) -> None:
    compilation = _compilation(tmp_path / "corpus")
    policy = _policy(compilation)
    prepared = ExternalRealTraceStabilityRunner().prepare_cases(
        compilation=compilation,
        policy=policy,
        kernel_audit=_audit(kernel=True),
    )[0]
    executor = ExternalRealTraceStabilityCaseExecutor()
    child_root = tmp_path / "child"

    first = executor.execute(
        prepared,
        child_root=child_root,
        audit=_audit(kernel=True),
    )
    replay = executor.execute(
        prepared,
        child_root=child_root,
        audit=_audit(kernel=True),
    )

    assert first.outcome is RealTraceStabilityCaseOutcomeV2.STABLE
    assert first.reason_code is RealTraceStabilityReasonCodeV2.NONE
    assert first.fault_observed is True
    assert first.resume_succeeded is True
    assert first.replay_stable is True
    assert first.attempt_count == 1
    assert first.unexpected_retry_count == 0
    assert first.resume_count == 1
    assert first.replay_count == 1
    assert replay.to_ref() == first.to_ref()


def test_external_stability_executes_curated_trajectory_adapter(
    tmp_path: Path,
) -> None:
    compilation = _curated_compilation(tmp_path / "curated")
    policy = _policy(compilation)
    prepared = ExternalRealTraceStabilityRunner().prepare_cases(
        compilation=compilation,
        policy=policy,
        kernel_audit=_audit(kernel=True),
    )[0]

    result = ExternalRealTraceStabilityCaseExecutor().execute(
        prepared,
        child_root=tmp_path / "child-curated",
        audit=_audit(kernel=True),
    )

    assert prepared.trace.adapter_name == "curated_trajectory_v1"
    assert result.outcome is RealTraceStabilityCaseOutcomeV2.STABLE
    assert result.resume_succeeded is True
    assert result.replay_stable is True


def test_external_stability_rejects_policy_inventory_drift(
    tmp_path: Path,
) -> None:
    compilation = _compilation(tmp_path / "corpus")
    policy = _policy(compilation)
    changed = policy.model_copy(update={"source_count": 99})

    with pytest.raises(
        ExternalRealTraceStabilityError,
        match="inventory",
    ):
        ExternalRealTraceStabilityRunner().prepare_cases(
            compilation=compilation,
            policy=changed,
            kernel_audit=_audit(kernel=True),
        )
