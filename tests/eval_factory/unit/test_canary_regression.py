from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
from test_canary_pipeline_driver import (
    _driver,
    _manifest,
    _with_storage_limit,
)
from test_canary_regression_builder import (
    CANARY_PATH,
    RAW_ROOT,
    _policy,
    _template,
)

from eval_factory.contracts.canary_execution_v2 import (
    r6_canary_execution_manifest_v2_ref,
)
from eval_factory.contracts.canary_regression_v2 import (
    CanaryRegressionCaseResultV2,
    CanaryRegressionExpectationV2,
    CanaryRegressionObservedOutcomeV2,
    CanaryRegressionOutcomeV2,
    CanaryRegressionReasonCodeV2,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef
from eval_factory.contracts.orchestration import ItemStatus, StageRunStatus
from eval_factory.contracts.orchestration_v2 import StageNameV2
from eval_factory.contracts.resource_v2 import ResourceAdmissionOutcomeV2
from eval_factory.orchestration.canary_driver import (
    CanaryPipelineDriver,
    create_canary_stage_store,
)
from eval_factory.orchestration.canary_errors import (
    CanaryProfileError,
    CanaryResourceAdmissionError,
)
from eval_factory.orchestration.job_store import (
    ImmutableResultError,
    JobStore,
    StaleWorkLeaseError,
)
from eval_factory.orchestration.lifecycle import PipelineLifecyclePolicyError
from eval_factory.readiness.canary_regression import (
    CanaryRegressionCaseEvidenceCompiler,
    CanaryRegressionRunConflictError,
    CanaryRegressionRunError,
    CanaryRegressionRunner,
    R6CanaryRegressionCaseExecutor,
    _admit_ledger,
    _ChildEvidence,
    _error_reason,
    _highest_stage,
    _internal_error_result,
    _private_root,
    _require_ledger,
    _terminal_reason,
    _validate_case_binding,
)
from eval_factory.readiness.canary_regression_builder import (
    CanaryRegressionBuilder,
    FrozenCanaryCohortLoader,
)
from eval_factory.readiness.canary_regression_models import (
    CanaryRegressionRunLedgerV1,
    FrozenCanaryCaseV1,
    PreparedCanaryRegressionCase,
)


def _prepared(
    driver: CanaryPipelineDriver,
    tmp_path: Path,
) -> PreparedCanaryRegressionCase:
    manifest = _manifest(tmp_path)
    trace = manifest.job_spec.traces[0]
    raw_path = Path(unquote(urlparse(trace.source_uri).path))
    case = FrozenCanaryCaseV1(
        instance_id="LH_005",
        source_ref="raw_traj://LH_005",
        raw_sha256=trace.raw_sha256,
        size_bytes=raw_path.stat().st_size,
        signal_quality="strict_events",
        verified_traits=("tool.file_read",),
        expectation=CanaryRegressionExpectationV2.MUST_SUCCEED,
    )
    del driver
    return PreparedCanaryRegressionCase(
        cohort_case=case,
        expectation=case.expectation,
        raw_path=raw_path,
        child_manifest=manifest,
    )


@pytest.mark.asyncio
async def test_evidence_compiler_accepts_complete_r6_success(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    prepared = _prepared(driver, tmp_path)
    child_result = await driver.run(prepared.child_manifest)

    result = CanaryRegressionCaseEvidenceCompiler().compile(
        prepared,
        job_store=driver.job_store,
        stage_store=driver.stage_store,
        child_result=child_result,
        error=None,
        audit=prepared.child_manifest.audit,
    )

    assert result.observed_outcome is CanaryRegressionObservedOutcomeV2.SUCCEEDED
    assert result.reason_code is CanaryRegressionReasonCodeV2.NONE
    assert result.highest_reached_stage is StageNameV2.ITEM_QUALITY
    assert result.expectation_matched is True
    assert result.work_unit_count == 10
    assert result.work_lease_count == 10
    assert result.stage_run_count == 9
    assert result.provider_invocation_count == 1


@pytest.mark.asyncio
async def test_evidence_compiler_maps_typed_resource_exit_without_message_parsing(
    tmp_path: Path,
) -> None:
    driver = _driver(tmp_path)
    prepared = _prepared(driver, tmp_path)
    limited = _with_storage_limit(
        prepared.child_manifest,
        storage_bytes=1,
    )
    prepared = PreparedCanaryRegressionCase(
        cohort_case=prepared.cohort_case,
        expectation=prepared.expectation,
        raw_path=prepared.raw_path,
        child_manifest=limited,
    )
    with pytest.raises(CanaryResourceAdmissionError) as caught:
        await driver.run(limited)
    assert caught.value.outcome is ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND

    result = CanaryRegressionCaseEvidenceCompiler().compile(
        prepared,
        job_store=driver.job_store,
        stage_store=driver.stage_store,
        child_result=None,
        error=caught.value,
        audit=limited.audit,
    )

    assert result.observed_outcome is CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR
    assert result.reason_code is CanaryRegressionReasonCodeV2.RESOURCE_UNSATISFIABLE
    assert "canary resource admission" not in result.model_dump_json()


def test_evidence_compiler_closes_absent_child_state_as_incomplete(
    tmp_path: Path,
) -> None:
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    driver = _driver(tmp_path / "unused")
    prepared = _prepared(driver, fixture_root)

    result = CanaryRegressionCaseEvidenceCompiler().compile(
        prepared,
        job_store=JobStore(tmp_path / "empty.sqlite3"),
        stage_store=create_canary_stage_store(tmp_path / "empty-stage"),
        child_result=None,
        error=None,
        audit=prepared.child_manifest.audit.model_copy(
            update={"created_by": "r8-03-incomplete-test"},
        ),
    )

    assert result.observed_outcome is CanaryRegressionObservedOutcomeV2.INCOMPLETE
    assert result.reason_code is CanaryRegressionReasonCodeV2.CHILD_STATE_INCOMPLETE
    assert result.highest_reached_stage is None


class _RecordingExecutor:
    def __init__(self) -> None:
        self.instance_ids: list[str] = []

    async def execute(
        self,
        prepared: PreparedCanaryRegressionCase,
        *,
        child_root: Path,
        audit: ContractAudit,
    ) -> CanaryRegressionCaseResultV2:
        del child_root
        instance_id = prepared.cohort_case.instance_id
        self.instance_ids.append(instance_id)
        if instance_id == "LH_020":
            raise RuntimeError("private injected detail")
        binding = prepared.child_manifest.trace_bindings[0]
        return CanaryRegressionCaseResultV2.create(
            instance_id=instance_id,
            source_trace_ref=binding.source_trace_ref,
            signal_quality=prepared.cohort_case.signal_quality,
            expectation=prepared.expectation,
            observed_outcome=(CanaryRegressionObservedOutcomeV2.INFRASTRUCTURE_ERROR),
            reason_code=CanaryRegressionReasonCodeV2.INTERNAL_ERROR,
            highest_reached_stage=None,
            child_manifest_ref=r6_canary_execution_manifest_v2_ref(prepared.child_manifest),
            child_dataset_result_ref=None,
            job_id=prepared.child_manifest.job_spec.job_id,
            job_status=None,
            item_id=None,
            item_status=None,
            stage_result_refs=(),
            quality_result_refs=(),
            package_manifest_refs=(),
            audit_report_ref=None,
            work_unit_count=0,
            work_lease_count=0,
            stage_run_count=0,
            attempt_count=0,
            retry_count=0,
            resume_count=0,
            provider_invocation_count=0,
            audit=audit,
        )


@pytest.mark.asyncio
async def test_runner_is_sequential_and_one_case_error_does_not_hide_later_cases(
    tmp_path: Path,
) -> None:
    policy = _policy()
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=policy)
    executor = _RecordingExecutor()

    report = await CanaryRegressionRunner(executor=executor).run(
        cohort=cohort,
        template=_template(tmp_path),
        policy=policy,
        raw_root=RAW_ROOT,
        run_root=tmp_path / "run",
        audit=policy.audit,
    )

    expected_ids = [case.instance_id for case in cohort.cases]
    assert executor.instance_ids == expected_ids
    assert [case.instance_id for case in report.case_results] == expected_ids
    assert report.outcome is CanaryRegressionOutcomeV2.INCOMPLETE
    assert len(report.infrastructure_error_ids) == 24
    assert "private injected detail" not in report.model_dump_json()


@pytest.mark.asyncio
async def test_real_repaired_heuristic_uses_final_child_audit_and_terminates(
    tmp_path: Path,
) -> None:
    policy = _policy()
    cohort = FrozenCanaryCohortLoader().load(CANARY_PATH, policy=policy)
    case = next(value for value in cohort.cases if value.instance_id == "LH_011")
    prepared = CanaryRegressionBuilder().prepare_case(
        case,
        template=_template(tmp_path),
        raw_root=RAW_ROOT,
        preparation_root=tmp_path / "preparation",
        policy=policy,
        audit=policy.audit,
    )

    result = await R6CanaryRegressionCaseExecutor().execute(
        prepared,
        child_root=tmp_path / "children" / case.instance_id,
        audit=policy.audit,
    )

    assert result.observed_outcome in {
        CanaryRegressionObservedOutcomeV2.SUCCEEDED,
        CanaryRegressionObservedOutcomeV2.BLOCKED,
        CanaryRegressionObservedOutcomeV2.FAILED,
    }
    assert result.expectation_matched is True


def _ref(object_type: str, suffix: str, *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r8-03-unit/{suffix}",
        object_version=version,
        object_sha256=(suffix.encode().hex() + ("a" * 64))[:64],
    )


def test_typed_evidence_and_error_matrices_are_closed(
    tmp_path: Path,
) -> None:
    manifest_ref = _ref("r6-canary-execution-manifest", "manifest")
    stage_ref = _ref("stage-result", "stage", version="record/v1")
    blocked = _ChildEvidence(
        item_status=ItemStatus.REJECTED,
        highest_reached_stage=StageNameV2.LABEL,
        stage_result_refs=(stage_ref,),
    )
    assert CanaryRegressionCaseEvidenceCompiler._classify(
        evidence=blocked,
        child_result=None,
        child_manifest_ref=manifest_ref,
        error=RuntimeError("ignored"),
    ) == (
        CanaryRegressionObservedOutcomeV2.BLOCKED,
        CanaryRegressionReasonCodeV2.LABEL_NO_MATCH,
    )

    for status, reason in (
        (
            StageRunStatus.BLOCKED_POLICY,
            CanaryRegressionReasonCodeV2.STAGE_BLOCKED_POLICY,
        ),
        (
            StageRunStatus.BLOCKED_CAPABILITY,
            CanaryRegressionReasonCodeV2.STAGE_BLOCKED_CAPABILITY,
        ),
        (
            StageRunStatus.TERMINAL_FAILURE,
            CanaryRegressionReasonCodeV2.STAGE_RETRY_EXHAUSTED,
        ),
    ):
        failed = _ChildEvidence(
            item_status=ItemStatus.FAILED,
            highest_reached_stage=StageNameV2.LABEL,
            terminal_stage_status=status,
            stage_result_refs=(stage_ref,),
        )
        assert CanaryRegressionCaseEvidenceCompiler._classify(
            evidence=failed,
            child_result=None,
            child_manifest_ref=manifest_ref,
            error=None,
        ) == (CanaryRegressionObservedOutcomeV2.FAILED, reason)
        assert _terminal_reason(status) is reason

    for error, reason in (
        (
            CanaryResourceAdmissionError(ResourceAdmissionOutcomeV2.WAITING_CAPACITY),
            CanaryRegressionReasonCodeV2.RESOURCE_WAITING,
        ),
        (
            CanaryResourceAdmissionError(ResourceAdmissionOutcomeV2.BUDGET_EXHAUSTED),
            CanaryRegressionReasonCodeV2.RESOURCE_BUDGET_EXHAUSTED,
        ),
        (
            CanaryProfileError("private"),
            CanaryRegressionReasonCodeV2.PROFILE_POLICY_ERROR,
        ),
        (
            PipelineLifecyclePolicyError("private"),
            CanaryRegressionReasonCodeV2.LIFECYCLE_POLICY_ERROR,
        ),
        (
            ImmutableResultError("private"),
            CanaryRegressionReasonCodeV2.MATERIAL_INTEGRITY_ERROR,
        ),
        (
            StaleWorkLeaseError("private"),
            CanaryRegressionReasonCodeV2.INTERNAL_ERROR,
        ),
        (
            RuntimeError("private"),
            CanaryRegressionReasonCodeV2.INTERNAL_ERROR,
        ),
    ):
        assert _error_reason(error) is reason

    assert _highest_stage(()) is None
    assert _highest_stage((StageNameV2.TRACE_INDEX, StageNameV2.LABEL)) is StageNameV2.LABEL

    root_file = tmp_path / "not-directory"
    root_file.write_text("x", encoding="utf-8")
    with pytest.raises(CanaryRegressionRunError, match="directory"):
        _private_root(root_file)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(CanaryRegressionRunError, match="symlink"):
        _private_root(link)


def test_run_ledger_and_case_binding_fail_closed(tmp_path: Path) -> None:
    children = tuple(_ref("r6-canary-execution-manifest", f"child-{index}") for index in range(24))
    ledger = CanaryRegressionRunLedgerV1.create(
        cohort_ref=_ref(
            "frozen-canary-cohort",
            "cohort",
            version="private-v1",
        ),
        template_ref=_ref(
            "canary-regression-template",
            "template",
            version="private-v1",
        ),
        policy_ref=_ref("canary-regression-policy", "policy"),
        child_manifest_refs=children,
    )
    root = _private_root(tmp_path / "run")
    _admit_ledger(root, ledger)
    _admit_ledger(root, ledger)
    _require_ledger(root / "run-ledger.json", ledger)

    changed = CanaryRegressionRunLedgerV1.create(
        cohort_ref=ledger.cohort_ref,
        template_ref=ledger.template_ref,
        policy_ref=ledger.policy_ref,
        child_manifest_refs=(
            *children[:-1],
            _ref("r6-canary-execution-manifest", "replacement"),
        ),
    )
    with pytest.raises(CanaryRegressionRunConflictError, match="conflicts"):
        _require_ledger(root / "run-ledger.json", changed)
    (root / "run-ledger.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CanaryRegressionRunConflictError, match="invalid"):
        _require_ledger(root / "run-ledger.json", ledger)

    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    prepared = _prepared(_driver(tmp_path / "unused"), fixture_root)
    internal = _internal_error_result(
        prepared,
        audit=prepared.child_manifest.audit,
    )
    _validate_case_binding(prepared, internal)
    crossed = internal.model_copy(update={"job_id": "job://crossed/identity"})
    with pytest.raises(CanaryRegressionRunError, match="crosses"):
        _validate_case_binding(prepared, crossed)
