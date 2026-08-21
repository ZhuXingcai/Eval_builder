from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.model_control_v2 import (
    R6_MODEL_CONTROL_POLICY_VERSION,
    JobModelPolicyV2,
    ModelAdmissionDecisionV2,
    ModelAdmissionOutcomeV2,
    ModelDemandModeV2,
    ModelRateDimensionV2,
    ModelRatePoolPolicyV2,
    ModelUsageSourceV2,
    ModelUsageV2,
    WorkModelDemandV2,
    WorkModelReservationV2,
    job_model_policy_v2_ref,
    model_admission_decision_v2_ref,
    model_rate_pool_policy_v2_ref,
    work_model_demand_v2_ref,
    work_model_reservation_v2_ref,
)

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _ref(object_type: str, suffix: str = "current", *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-03/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _audit(*refs: ObjectRef, created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r6-03-contract-test",
        governing_versions=(
            VersionBinding(
                component="model-control",
                version=R6_MODEL_CONTROL_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(refs, key=_ref_key)),
    )


def _pool(*, created_at: datetime = NOW) -> ModelRatePoolPolicyV2:
    bucket_ref = _ref("provider-rate-bucket", version="v1")
    profile_ref = _ref("model-profile", version="v1")
    return ModelRatePoolPolicyV2.create(
        provider_bucket_ref=bucket_ref,
        allowed_model_profile_refs=(profile_ref,),
        requests_per_minute=60,
        tokens_per_minute=60_000,
        request_burst=4,
        token_burst=8_000,
        max_concurrent_requests=2,
        minimum_backpressure_seconds=5,
        audit=_audit(bucket_ref, profile_ref, created_at=created_at),
    )


def _job_policy(pool: ModelRatePoolPolicyV2) -> JobModelPolicyV2:
    spec_ref = _ref("dataset-job-spec")
    graph_ref = _ref("resolved-job-work-graph")
    profile_ref = pool.allowed_model_profile_refs[0]
    pool_ref = model_rate_pool_policy_v2_ref(pool)
    return JobModelPolicyV2.create(
        dataset_job_spec_ref=spec_ref,
        resolved_job_work_graph_ref=graph_ref,
        model_rate_pool_policy_refs=(pool_ref,),
        model_profile_refs=(profile_ref,),
        max_concurrent_requests=1,
        max_model_requests=8,
        max_model_tokens=20_000,
        audit=_audit(spec_ref, graph_ref, pool_ref, profile_ref),
    )


def _demand(policy: JobModelPolicyV2) -> WorkModelDemandV2:
    unit_ref = _ref("resolved-work-unit")
    operation_ref = _ref("semantic-residual-request")
    execution_profile_ref = _ref("model-execution-profile", version="v1")
    policy_ref = job_model_policy_v2_ref(policy)
    pool_ref = policy.model_rate_pool_policy_refs[0]
    profile_ref = policy.model_profile_refs[0]
    return WorkModelDemandV2.create(
        resolved_job_work_graph_ref=policy.resolved_job_work_graph_ref,
        work_unit_ref=unit_ref,
        job_model_policy_ref=policy_ref,
        model_rate_pool_policy_ref=pool_ref,
        model_profile_ref=profile_ref,
        operation_ref=operation_ref,
        model_execution_profile_ref=execution_profile_ref,
        attempt=1,
        mode=ModelDemandModeV2.DIRECT_REQUEST,
        request_allowance=1,
        input_token_allowance=1000,
        output_token_allowance=2000,
        cache_token_allowance=500,
        concurrency_units=1,
        audit=_audit(
            policy.resolved_job_work_graph_ref,
            unit_ref,
            policy_ref,
            pool_ref,
            profile_ref,
            operation_ref,
            execution_profile_ref,
        ),
    )


def test_model_policy_contracts_are_strict_frozen_and_audit_time_independent() -> None:
    first = _pool()
    second = _pool(created_at=NOW - timedelta(days=1))

    assert first.model_rate_pool_policy_id == second.model_rate_pool_policy_id
    assert first.model_rate_pool_policy_sha256 == second.model_rate_pool_policy_sha256
    assert model_rate_pool_policy_v2_ref(first).object_sha256 == (first.model_rate_pool_policy_sha256)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ModelRatePoolPolicyV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "api_endpoint": "https://private.invalid",
            }
        )
    with pytest.raises(ValidationError, match="frozen"):
        first.request_burst = 99  # type: ignore[misc]


def test_model_demand_binds_allowance_components_and_mode() -> None:
    demand = _demand(_job_policy(_pool()))

    assert demand.token_allowance == 3500
    assert work_model_demand_v2_ref(demand).object_sha256 == demand.work_model_demand_sha256
    with pytest.raises(ValidationError, match="token allowance"):
        WorkModelDemandV2.model_validate(
            {
                **demand.model_dump(mode="python"),
                "token_allowance": 3499,
            }
        )
    with pytest.raises(ValidationError, match="DIRECT_REQUEST"):
        WorkModelDemandV2.model_validate(
            {
                **demand.model_dump(mode="python"),
                "request_allowance": 2,
            }
        )


def test_usage_requires_exact_total_and_explicit_source() -> None:
    usage = ModelUsageV2(
        requests=1,
        input_tokens=10,
        output_tokens=20,
        cache_creation_input_tokens=2,
        cache_read_input_tokens=3,
        charged_tokens=35,
        source=ModelUsageSourceV2.REPORTED,
    )

    assert usage.charged_tokens == 35
    with pytest.raises(ValidationError, match="charged_tokens"):
        ModelUsageV2.model_validate(
            {
                **usage.model_dump(mode="python"),
                "charged_tokens": 34,
            }
        )


@pytest.mark.parametrize(
    ("outcome", "dimensions", "eligible"),
    [
        (
            ModelAdmissionOutcomeV2.WAITING_RATE,
            (ModelRateDimensionV2.REQUESTS,),
            NOW + timedelta(seconds=1),
        ),
        (
            ModelAdmissionOutcomeV2.WAITING_PROVIDER,
            (ModelRateDimensionV2.PROVIDER_BACKPRESSURE,),
            NOW + timedelta(seconds=5),
        ),
        (
            ModelAdmissionOutcomeV2.WAITING_CONCURRENCY,
            (ModelRateDimensionV2.CONCURRENCY,),
            None,
        ),
        (
            ModelAdmissionOutcomeV2.BUDGET_EXHAUSTED,
            (ModelRateDimensionV2.TOKENS,),
            None,
        ),
        (
            ModelAdmissionOutcomeV2.UNSATISFIABLE_DEMAND,
            (ModelRateDimensionV2.REQUESTS,),
            None,
        ),
    ],
)
def test_negative_model_admission_has_closed_eligibility_matrix(
    outcome: ModelAdmissionOutcomeV2,
    dimensions: tuple[ModelRateDimensionV2, ...],
    eligible: datetime | None,
) -> None:
    demand = _demand(_job_policy(_pool()))
    policy_ref = demand.job_model_policy_ref
    demand_ref = work_model_demand_v2_ref(demand)
    decision = ModelAdmissionDecisionV2.create(
        job_model_policy_ref=policy_ref,
        work_model_demand_ref=demand_ref,
        outcome=outcome,
        constrained_dimensions=dimensions,
        eligible_at=eligible,
        pool_head_version_before=0,
        job_head_version_before=0,
        decided_at=NOW,
        audit=_audit(policy_ref, demand_ref),
    )

    assert model_admission_decision_v2_ref(decision).object_sha256 == (
        decision.model_admission_decision_sha256
    )


def test_admitted_reservation_binds_one_lease_and_fence() -> None:
    demand = _demand(_job_policy(_pool()))
    policy_ref = demand.job_model_policy_ref
    demand_ref = work_model_demand_v2_ref(demand)
    decision = ModelAdmissionDecisionV2.create(
        job_model_policy_ref=policy_ref,
        work_model_demand_ref=demand_ref,
        outcome=ModelAdmissionOutcomeV2.ADMITTED,
        constrained_dimensions=(),
        eligible_at=None,
        pool_head_version_before=0,
        job_head_version_before=0,
        decided_at=NOW,
        audit=_audit(policy_ref, demand_ref),
    )
    decision_ref = model_admission_decision_v2_ref(decision)
    lease_ref = _ref("work-lease")
    dispatch_ref = _ref("work-dispatch-decision")
    holder_ref = _ref("worker-principal", version="v1")
    handle_ref = _ref("model-invocation-handle", version="v1")
    reservation = WorkModelReservationV2.create(
        model_admission_decision_ref=decision_ref,
        work_model_demand_ref=demand_ref,
        work_lease_ref=lease_ref,
        work_dispatch_decision_ref=dispatch_ref,
        holder_ref=holder_ref,
        fencing_token=3,
        invocation_handle_ref=handle_ref,
        request_allowance=demand.request_allowance,
        token_allowance=demand.token_allowance,
        concurrency_units=demand.concurrency_units,
        acquired_at=NOW,
        audit=_audit(
            decision_ref,
            demand_ref,
            lease_ref,
            dispatch_ref,
            holder_ref,
            handle_ref,
        ),
    )

    assert reservation.fencing_token == 3
    assert work_model_reservation_v2_ref(reservation).object_sha256 == (
        reservation.work_model_reservation_sha256
    )
