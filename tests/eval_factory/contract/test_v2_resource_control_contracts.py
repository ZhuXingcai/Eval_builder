from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding
from eval_factory.contracts.resource_v2 import (
    R6_RESOURCE_CONTROL_POLICY_VERSION,
    JobResourcePolicyV2,
    NonModelResourceVectorV2,
    ResourceAdmissionDecisionV2,
    ResourceAdmissionOutcomeV2,
    ResourceKindV2,
    ResourcePoolPolicyV2,
    WorkResourceDemandV2,
    WorkResourceReservationV2,
    job_resource_policy_v2_ref,
    resource_admission_decision_v2_ref,
    resource_pool_policy_v2_ref,
    work_resource_demand_v2_ref,
    work_resource_reservation_v2_ref,
)

HASH = "a" * 64
NOW = datetime(2026, 7, 31, tzinfo=UTC)


def _ref(object_type: str, suffix: str = "current", *, version: str = "v2") -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://r6-04/{suffix}",
        object_version=version,
        object_sha256=HASH,
    )


def _audit(*refs: ObjectRef, created_at: datetime = NOW) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="r6-04-contract-test",
        governing_versions=(
            VersionBinding(
                component="resource-control",
                version=R6_RESOURCE_CONTROL_POLICY_VERSION,
            ),
        ),
        input_refs=tuple(sorted(refs, key=_ref_key)),
    )


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _vector(
    *,
    processes: int = 1,
    renderers: int = 0,
    network_requests: int = 0,
    storage_bytes: int = 1024,
) -> NonModelResourceVectorV2:
    return NonModelResourceVectorV2(
        processes=processes,
        renderers=renderers,
        network_requests=network_requests,
        storage_bytes=storage_bytes,
    )


def _pool(*, created_at: datetime = NOW) -> ResourcePoolPolicyV2:
    domain_ref = _ref("resource-domain", version="v1")
    return ResourcePoolPolicyV2.create(
        resource_domain_ref=domain_ref,
        capacity=_vector(processes=2, renderers=1, network_requests=3, storage_bytes=4096),
        audit=_audit(domain_ref, created_at=created_at),
    )


def _job_policy(pool: ResourcePoolPolicyV2) -> JobResourcePolicyV2:
    spec_ref = _ref("dataset-job-spec")
    graph_ref = _ref("resolved-job-work-graph")
    pool_ref = resource_pool_policy_v2_ref(pool)
    return JobResourcePolicyV2.create(
        dataset_job_spec_ref=spec_ref,
        resolved_job_work_graph_ref=graph_ref,
        resource_pool_policy_ref=pool_ref,
        job_concurrency=_vector(processes=1, renderers=1, network_requests=1, storage_bytes=4096),
        job_budget=_vector(processes=2, renderers=1, network_requests=3, storage_bytes=4096),
        audit=_audit(spec_ref, graph_ref, pool_ref),
    )


def _demand(policy: JobResourcePolicyV2) -> WorkResourceDemandV2:
    graph_ref = policy.resolved_job_work_graph_ref
    unit_ref = _ref("resolved-work-unit")
    policy_ref = job_resource_policy_v2_ref(policy)
    profile_ref = _ref("resource-execution-profile", version="v1")
    return WorkResourceDemandV2.create(
        resolved_job_work_graph_ref=graph_ref,
        work_unit_ref=unit_ref,
        job_resource_policy_ref=policy_ref,
        execution_profile_ref=profile_ref,
        attempt=1,
        capacity_units=_vector(),
        budget_allowance=_vector(),
        termination_required=True,
        audit=_audit(graph_ref, unit_ref, policy_ref, profile_ref),
    )


def test_resource_contracts_are_strict_frozen_and_audit_time_independent() -> None:
    first = _pool()
    second = _pool(created_at=NOW.replace(day=30))

    assert first.resource_pool_policy_id == second.resource_pool_policy_id
    assert first.resource_pool_policy_sha256 == second.resource_pool_policy_sha256
    assert resource_pool_policy_v2_ref(first).object_sha256 == first.resource_pool_policy_sha256
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ResourcePoolPolicyV2.model_validate(
            {
                **first.model_dump(mode="python"),
                "physical_path": "/private/runtime",
            }
        )
    with pytest.raises(ValidationError, match="frozen"):
        first.capacity = _vector(processes=9)  # type: ignore[misc]


def test_process_demand_requires_termination_and_storage_capacity_matches_allowance() -> None:
    policy = _job_policy(_pool())
    demand = _demand(policy)

    assert work_resource_demand_v2_ref(demand).object_sha256 == demand.work_resource_demand_sha256
    with pytest.raises(ValidationError, match="termination"):
        WorkResourceDemandV2.model_validate(
            {
                **demand.model_dump(mode="python"),
                "termination_required": False,
            }
        )
    with pytest.raises(ValidationError, match="storage"):
        WorkResourceDemandV2.model_validate(
            {
                **demand.model_dump(mode="python"),
                "budget_allowance": _vector(storage_bytes=2048),
            }
        )
    with pytest.raises(ValidationError, match="active capacity"):
        WorkResourceDemandV2.model_validate(
            {
                **demand.model_dump(mode="python"),
                "capacity_units": _vector(processes=0),
            }
        )


@pytest.mark.parametrize(
    ("outcome", "constrained"),
    [
        (ResourceAdmissionOutcomeV2.WAITING_CAPACITY, (ResourceKindV2.PROCESS,)),
        (ResourceAdmissionOutcomeV2.BUDGET_EXHAUSTED, (ResourceKindV2.STORAGE,)),
        (ResourceAdmissionOutcomeV2.UNSATISFIABLE_DEMAND, (ResourceKindV2.RENDERER,)),
    ],
)
def test_negative_admission_is_content_free_and_cannot_create_reservation(
    outcome: ResourceAdmissionOutcomeV2,
    constrained: tuple[ResourceKindV2, ...],
) -> None:
    demand = _demand(_job_policy(_pool()))
    demand_ref = work_resource_demand_v2_ref(demand)
    policy_ref = demand.job_resource_policy_ref
    decision = ResourceAdmissionDecisionV2.create(
        job_resource_policy_ref=policy_ref,
        work_resource_demand_ref=demand_ref,
        outcome=outcome,
        constrained_resources=constrained,
        pool_head_version_before=0,
        job_head_version_before=0,
        decided_at=NOW,
        audit=_audit(policy_ref, demand_ref),
    )

    assert resource_admission_decision_v2_ref(decision).object_sha256 == (
        decision.resource_admission_decision_sha256
    )
    with pytest.raises(ValidationError, match="ADMITTED"):
        WorkResourceReservationV2.create(
            resource_admission_decision_ref=resource_admission_decision_v2_ref(decision),
            work_resource_demand_ref=demand_ref,
            work_lease_ref=_ref("work-lease"),
            work_dispatch_decision_ref=_ref("work-dispatch-decision"),
            holder_ref=_ref("worker-principal", version="v1"),
            fencing_token=1,
            execution_handle_ref=_ref("resource-execution-handle", version="v1"),
            capacity_units=demand.capacity_units,
            budget_allowance=demand.budget_allowance,
            acquired_at=NOW,
            audit=_audit(resource_admission_decision_v2_ref(decision), demand_ref),
            admission_outcome=outcome,
        )


def test_admitted_decision_and_reservation_bind_exact_lease_and_fence() -> None:
    demand = _demand(_job_policy(_pool()))
    demand_ref = work_resource_demand_v2_ref(demand)
    policy_ref = demand.job_resource_policy_ref
    decision = ResourceAdmissionDecisionV2.create(
        job_resource_policy_ref=policy_ref,
        work_resource_demand_ref=demand_ref,
        outcome=ResourceAdmissionOutcomeV2.ADMITTED,
        constrained_resources=(),
        pool_head_version_before=0,
        job_head_version_before=0,
        decided_at=NOW,
        audit=_audit(policy_ref, demand_ref),
    )
    decision_ref = resource_admission_decision_v2_ref(decision)
    lease_ref = _ref("work-lease")
    dispatch_ref = _ref("work-dispatch-decision")
    holder_ref = _ref("worker-principal", version="v1")
    handle_ref = _ref("resource-execution-handle", version="v1")
    reservation = WorkResourceReservationV2.create(
        resource_admission_decision_ref=decision_ref,
        work_resource_demand_ref=demand_ref,
        work_lease_ref=lease_ref,
        work_dispatch_decision_ref=dispatch_ref,
        holder_ref=holder_ref,
        fencing_token=1,
        execution_handle_ref=handle_ref,
        capacity_units=demand.capacity_units,
        budget_allowance=demand.budget_allowance,
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

    assert work_resource_reservation_v2_ref(reservation).object_sha256 == (
        reservation.work_resource_reservation_sha256
    )
