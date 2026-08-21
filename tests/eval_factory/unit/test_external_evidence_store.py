from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from eval_factory.contracts.core import (
    ContractAudit,
    EvidencePolarity,
    EvidenceRef,
    ObjectRef,
    SourceSpanRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import (
    LabelDecisionV2,
    LabelDecisionValueV2,
    LabelExecutionStatus,
)
from eval_factory.labeling.decision import LabelDecisionRoute
from eval_factory.readiness.external_evidence_models import (
    ExternalCorpusInventoryV1,
    ExternalCorpusMemberV1,
    ExternalEvidenceMaterialClosureV1,
    ExternalPartitionInventoryV1,
    ExternalPartitionKindV1,
    ExternalPartitionMemberV1,
    ExternalRuntimeV1,
)
from eval_factory.readiness.external_evidence_store import (
    ExternalEvidenceMaterialStore,
    ExternalEvidenceStoreFaultPoint,
    ExternalEvidenceStoreInjectedCrash,
    ExternalEvidenceStoreIntegrityError,
    ExternalEvidenceStoreLimitError,
    ExternalEvidenceStoreTypeError,
    StaticExternalEvidenceStoreFaultInjector,
)
from eval_factory.readiness.external_observation_models import (
    BlindLabelObservationRecordV1,
    BlindLabelObservationSetV1,
)
from eval_factory.readiness.external_reference_models import (
    ExternalReferenceAuthorKindV1,
    ExternalReferenceRecordV1,
    ExternalReferenceSetV1,
    ExternalReferenceStateV1,
)

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=NOW,
        created_by="external-store-test",
        governing_versions=(
            VersionBinding(
                component="external-evidence-store",
                version="external-evidence/r8-10-v1",
            ),
        ),
    )


def _authorization_ref() -> ObjectRef:
    return ObjectRef(
        object_type="external-source-authorization",
        object_id="external-source-authorization://store-tests/source",
        object_version="v2",
        object_sha256=_digest("external-store-source-authorization"),
    )


def _inventory(
    *,
    count: int = 100,
) -> ExternalCorpusInventoryV1:
    runtimes = (
        ExternalRuntimeV1.CLAUDE_CODE,
        ExternalRuntimeV1.CODEX,
        ExternalRuntimeV1.HERMES,
    )
    api_types = {
        ExternalRuntimeV1.CLAUDE_CODE: "Message",
        ExternalRuntimeV1.CODEX: "Response",
        ExternalRuntimeV1.HERMES: "Chat",
    }
    members = tuple(
        ExternalCorpusMemberV1.create(
            runtime=(runtime := runtimes[index % len(runtimes)]),
            relative_name=(f"{runtime.value}/req_{index:016x}_raw.json"),
            sid=f"sid-{index:04d}",
            raw_sha256=_digest(f"raw:{index}"),
            size_bytes=1_000 + index,
            event_time="2026-07-17 00:00:00",
            api_type=api_types[runtime],
            business="CodingPlan",
            real_model=f"{runtime.value}-real",
            request_model=f"{runtime.value}-request",
        )
        for index in range(count)
    )
    return ExternalCorpusInventoryV1.create(
        source_authorization_ref=_authorization_ref(),
        members=members,
        audit=_audit(),
    )


def _partition(
    inventory: ExternalCorpusInventoryV1,
) -> ExternalPartitionInventoryV1:
    return ExternalPartitionInventoryV1.create(
        partition_kind=ExternalPartitionKindV1.TEST,
        members=tuple(
            ExternalPartitionMemberV1(
                source_trace_id=member.source_trace_id,
                raw_sha256=member.raw_sha256,
            )
            for member in inventory.members
        ),
        source_authority_refs=(
            inventory.to_ref(),
            inventory.to_source_population_ref(),
        ),
        audit=_audit(),
    )


def _reference_set(
    inventory: ExternalCorpusInventoryV1,
) -> ExternalReferenceSetV1:
    member = inventory.members[0]
    trace_envelope_ref = ObjectRef(
        object_type="trace-envelope",
        object_id="trace-envelope://store-tests/trace",
        object_version="v1",
        object_sha256=_digest("trace-envelope"),
    )
    label_spec_ref = ObjectRef(
        object_type="label-spec",
        object_id="label-spec://store-tests/search",
        object_version="v2",
        object_sha256=_digest("label-spec"),
    )
    annotation_contract_ref = ObjectRef(
        object_type="annotation-contract-manifest",
        object_id="annotation-contract-manifest://store-tests/v1",
        object_version="v1",
        object_sha256=_digest("annotation-contract"),
    )
    reference_policy_ref = ObjectRef(
        object_type="external-reference-authoring-policy",
        object_id="external-reference-authoring-policy://store-tests/v2",
        object_version="v2",
        object_sha256=_digest("reference-policy"),
    )
    record = ExternalReferenceRecordV1.create(
        source_trace_id=member.source_trace_id,
        raw_sha256=member.raw_sha256,
        trace_envelope_ref=trace_envelope_ref,
        label_spec_ref=label_spec_ref,
        label_name="search_tool_usage",
        annotation_contract_ref=annotation_contract_ref,
        reference_policy_ref=reference_policy_ref,
        expected_decision=LabelDecisionValueV2.NO_MATCH,
        evidence_refs=(trace_envelope_ref,),
        structured_capability_complete=True,
        author_kind=ExternalReferenceAuthorKindV1.DETERMINISTIC_SERVICE,
        reference_state=ExternalReferenceStateV1.REFERENCE_READY,
        rule_version="external-structured-reference/r8-10-v1",
        model_profile=None,
        prompt_version=None,
        audit=_audit(),
    )
    return ExternalReferenceSetV1.create(
        source_population_ref=inventory.to_source_population_ref(),
        label_spec_refs=(label_spec_ref,),
        reference_policy_ref=reference_policy_ref,
        records=(record,),
        audit=_audit(),
    )


def _observation_set(
    inventory: ExternalCorpusInventoryV1,
) -> BlindLabelObservationSetV1:
    member = inventory.members[0]
    trace_envelope_ref = ObjectRef(
        object_type="trace-envelope",
        object_id="trace-envelope://store-tests/observation",
        object_version="v1",
        object_sha256=_digest("observation-trace-envelope"),
    )
    label_spec_ref = ObjectRef(
        object_type="label-spec",
        object_id="label-spec://store-tests/observation",
        object_version="v2",
        object_sha256=_digest("observation-label-spec"),
    )
    observation_policy_ref = ObjectRef(
        object_type="external-observation-policy",
        object_id="external-observation-policy://store-tests/v2",
        object_version="v2",
        object_sha256=_digest("observation-policy"),
    )
    evidence = EvidenceRef(
        evidence_ref_id="evidence-ref://store-tests/negative",
        subject_ref=trace_envelope_ref,
        source_spans=(
            SourceSpanRef(
                span_id="source-span://store-tests/negative",
                source_trace_id=member.source_trace_id,
                raw_sha256=member.raw_sha256,
            ),
        ),
        polarity=EvidencePolarity.NEGATIVE,
        capability="tool_events",
        capability_complete=True,
    )
    decision = LabelDecisionV2(
        label_decision_id="label-decision://store-tests/negative",
        label_spec_ref=label_spec_ref,
        trace_envelope_ref=trace_envelope_ref,
        decision=LabelDecisionValueV2.NO_MATCH,
        execution_status=LabelExecutionStatus.FINAL,
        positive_evidence=(),
        negative_evidence=(evidence,),
        semantic_evidence=(),
        structured_capability_complete=True,
        confidence=1.0,
        rule_version="structured-labeling/r3-02-v1",
        model_profile=None,
        prompt_version=None,
        unresolved_reasons=frozenset(),
        policy_version="label-decision-merge/r3-04-v1",
        decision_sha256=_digest("observation-decision"),
        audit=_audit(),
    )
    record = BlindLabelObservationRecordV1.create(
        source_trace_id=member.source_trace_id,
        raw_sha256=member.raw_sha256,
        trace_envelope_ref=trace_envelope_ref,
        label_spec_ref=label_spec_ref,
        observation_policy_ref=observation_policy_ref,
        label_decision=decision,
        route=LabelDecisionRoute.FINAL,
        audit=_audit(),
    )
    return BlindLabelObservationSetV1.create(
        source_population_ref=inventory.to_source_population_ref(),
        label_spec_refs=(label_spec_ref,),
        observation_policy_ref=observation_policy_ref,
        records=(record,),
        audit=_audit(),
    )


def _store(
    root: Path,
    *,
    fault_injector: StaticExternalEvidenceStoreFaultInjector | None = None,
    max_private_bytes: int = 10_000_000,
    max_members: int = 1_000,
) -> ExternalEvidenceMaterialStore:
    return ExternalEvidenceMaterialStore(
        root,
        max_private_bytes=max_private_bytes,
        max_members=max_members,
        fault_injector=fault_injector,
    )


def test_store_round_trip_and_exact_replay(tmp_path: Path) -> None:
    inventory = _inventory()
    partition = _partition(inventory)
    train = ExternalPartitionInventoryV1.create(
        partition_kind=ExternalPartitionKindV1.TRAIN,
        members=(),
        source_authority_refs=(inventory.to_ref(),),
        audit=_audit(),
    )
    development = ExternalPartitionInventoryV1.create(
        partition_kind=ExternalPartitionKindV1.DEVELOPMENT,
        members=(),
        source_authority_refs=(inventory.to_ref(),),
        audit=_audit(),
    )
    reference_set = _reference_set(inventory)
    observation_set = _observation_set(inventory)
    closure = ExternalEvidenceMaterialClosureV1.create(
        inventory_ref=inventory.to_ref(),
        partition_refs=(
            train.to_ref(),
            development.to_ref(),
            partition.to_ref(),
        ),
        reference_set_ref=reference_set.to_ref(),
        observation_set_ref=observation_set.to_ref(),
        source_count=inventory.source_count,
        total_pair_count=2,
        material_canonical_bytes=sum(
            len(value.canonical_json())
            for value in (
                inventory,
                train,
                development,
                partition,
                reference_set,
                observation_set,
            )
        ),
        audit=_audit(),
    )
    store = _store(tmp_path / "private")

    inventory_write = store.put_inventory(inventory)
    store.put_partition(train)
    store.put_partition(development)
    partition_write = store.put_partition(partition)
    reference_write = store.put_reference_set(reference_set)
    observation_write = store.put_observation_set(observation_set)
    closure_write = store.put_material_closure(closure)
    inventory_replay = store.put_inventory(inventory)
    partition_replay = store.put_partition(partition)
    reference_replay = store.put_reference_set(reference_set)
    observation_replay = store.put_observation_set(observation_set)
    closure_replay = store.put_material_closure(closure)

    assert inventory_write.written is True
    assert partition_write.written is True
    assert reference_write.written is True
    assert observation_write.written is True
    assert closure_write.written is True
    assert inventory_replay.written is False
    assert partition_replay.written is False
    assert reference_replay.written is False
    assert observation_replay.written is False
    assert closure_replay.written is False
    assert store.get_inventory(inventory.to_ref()) == inventory
    assert store.get_partition(partition.to_ref()) == partition
    assert store.get_reference_set(reference_set.to_ref()) == reference_set
    assert store.get_observation_set(observation_set.to_ref()) == (observation_set)
    assert store.get_material_closure(closure.to_ref()) == closure
    assert not hasattr(store, "list")
    assert not hasattr(store, "path_for")


def test_store_rejects_wrong_type_and_limits(tmp_path: Path) -> None:
    inventory = _inventory()
    partition = _partition(inventory)
    store = _store(tmp_path / "private")
    store.put_inventory(inventory)

    with pytest.raises(ExternalEvidenceStoreTypeError):
        store.get_partition(inventory.to_ref())
    with pytest.raises(ExternalEvidenceStoreLimitError):
        _store(
            tmp_path / "limited-members",
            max_members=99,
        ).put_inventory(inventory)
    with pytest.raises(ExternalEvidenceStoreLimitError):
        _store(
            tmp_path / "limited-bytes",
            max_private_bytes=2,
        ).put_partition(partition)


@pytest.mark.parametrize(
    "fault_point",
    tuple(ExternalEvidenceStoreFaultPoint),
)
def test_store_fault_replay_recovers_without_duplicate_authority(
    tmp_path: Path,
    fault_point: ExternalEvidenceStoreFaultPoint,
) -> None:
    inventory = _inventory()
    root = tmp_path / fault_point.value
    faulting = _store(
        root,
        fault_injector=StaticExternalEvidenceStoreFaultInjector(crash_points=frozenset({fault_point})),
    )

    with pytest.raises(ExternalEvidenceStoreInjectedCrash):
        faulting.put_inventory(inventory)

    clean = _store(root)
    write = clean.put_inventory(inventory)
    assert clean.get_inventory(inventory.to_ref()) == inventory
    if fault_point is ExternalEvidenceStoreFaultPoint.AFTER_CAS_WRITE:
        assert write.written is True
        assert write.content_blob_written is False
    else:
        assert write.written is False


def test_store_detects_envelope_and_content_corruption(
    tmp_path: Path,
) -> None:
    inventory = _inventory()
    root = tmp_path / "private"
    store = _store(root)
    store.put_inventory(inventory)

    envelopes = tuple((root / "envelopes").rglob("*.json"))
    assert len(envelopes) == 1
    envelopes[0].write_text("{}", encoding="utf-8")
    with pytest.raises(ExternalEvidenceStoreIntegrityError):
        store.get_inventory(inventory.to_ref())

    content_root = tmp_path / "content"
    content_store = _store(content_root)
    content_store.put_inventory(inventory)
    blobs = tuple((content_root / "cas" / "sha256").rglob("*"))
    blob = next(path for path in blobs if path.is_file())
    blob.write_bytes(b"corrupt")
    with pytest.raises(ExternalEvidenceStoreIntegrityError):
        content_store.get_inventory(inventory.to_ref())


def test_store_rejects_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    with pytest.raises(ExternalEvidenceStoreTypeError):
        _store(linked)
