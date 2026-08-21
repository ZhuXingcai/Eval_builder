from __future__ import annotations

import inspect
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from eval_factory.contracts.core import (
    ContractAudit,
    ObjectRef,
    VersionBinding,
)
from eval_factory.contracts.labeling_v2 import LabelDecisionValueV2
from eval_factory.contracts.trace import (
    CapabilityStatus,
    ParseQuality,
    TraceCapability,
)
from eval_factory.readiness.external_blind_observation import (
    ExternalBlindStructuredObserver,
)
from eval_factory.readiness.external_observation_models import (
    BlindLabelObservationSetV1,
)
from eval_factory.readiness.external_reference_authoring import (
    APPROVED_EXTERNAL_LABEL_NAMES,
    ExternalStructuredReferenceBuilder,
    approved_external_label_specs,
    external_trace_envelope_ref,
)
from eval_factory.readiness.external_reference_models import (
    ExternalReferenceAuthorKindV1,
    ExternalReferenceStateV1,
)
from eval_factory.trace import (
    RawTrajRecovery,
    RawTrajV1Normalizer,
    RuntimeSnapshotV1Adapter,
    RuntimeSnapshotV1Parser,
    TraceIndexBuilder,
    TraceSourceRegistry,
)
from eval_factory.trace.indexing.models import TraceIndexResult


def _audit(
    created_at: datetime = datetime(2026, 8, 9, tzinfo=UTC),
) -> ContractAudit:
    return ContractAudit(
        created_at=created_at,
        created_by="external-reference-test",
        governing_versions=(
            VersionBinding(
                component="external-reference-authoring",
                version="external-reference/r8-10-v1",
            ),
        ),
    )


def _ref(
    object_type: str,
    suffix: str,
    *,
    version: str = "v1",
) -> ObjectRef:
    import hashlib

    digest = hashlib.sha256(f"{object_type}:{suffix}:{version}".encode()).hexdigest()
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://tests/{suffix}",
        object_version=version,
        object_sha256=digest,
    )


def _write_snapshot(
    path: Path,
    *,
    content: list[dict[str, object]],
) -> None:
    outer = {
        "sid": "reference-session",
        "event_time": "2026-07-17 00:00:08",
        "api_type": "Message",
        "business": "CodingPlan",
        "real_model": "model-a",
        "request_model": "model-request-a",
        "request": json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "inspect and recover"},
                    {"role": "assistant", "content": content},
                ]
            },
            separators=(",", ":"),
        ),
        "response": json.dumps(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "continuing"}],
            },
            separators=(",", ":"),
        ),
    }
    path.write_text(
        json.dumps(outer, separators=(",", ":")),
        encoding="utf-8",
    )


def _index(
    source: Path,
    root: Path,
) -> TraceIndexResult:
    adapter = RuntimeSnapshotV1Adapter()
    registered = TraceSourceRegistry(
        root / "registry.sqlite3",
        adapter=adapter,
    ).register(
        source,
        source_trace_id="source-trace://external/reference-test",
        source_uri="runtime-snapshot://reference-test/source",
    )
    parsed = RuntimeSnapshotV1Parser().parse(
        source,
        registered_source=registered,
        audit=_audit(),
    )
    recovered = RawTrajRecovery(adapter=adapter).recover(
        source,
        parse_result=parsed,
        audit=_audit(),
    )
    normalized = RawTrajV1Normalizer(adapter=adapter).normalize(
        source,
        recovery_result=recovered,
        audit=_audit(),
    )
    return TraceIndexBuilder().build(
        normalization_result=normalized,
        audit=_audit(),
    )


def _builder() -> ExternalStructuredReferenceBuilder:
    return ExternalStructuredReferenceBuilder(
        annotation_contract_ref=_ref(
            "annotation-contract-manifest",
            "annotation-v2",
        ),
        reference_policy_ref=_ref(
            "external-reference-authoring-policy",
            "structured",
            version="v2",
        ),
    )


def test_approved_label_portfolio_is_exact_and_stable() -> None:
    first = approved_external_label_specs(audit=_audit())
    second = approved_external_label_specs(audit=_audit(datetime(2026, 8, 10, tzinfo=UTC)))
    by_name = {spec.name: spec for spec in first}

    assert tuple(spec.name for spec in first) == APPROVED_EXTERNAL_LABEL_NAMES
    assert sum(spec.semantic_residual is None for spec in first) == 2
    assert sum(spec.semantic_residual is not None for spec in first) == 1
    assert tuple(spec.label_spec_sha256 for spec in second) == tuple(spec.label_spec_sha256 for spec in first)
    powershell = by_name["powershell_error_signature"]
    assert powershell.label_spec_id == ("label-spec://powershell-error-signature/r8-10-v2")
    assert powershell.label_version == "v2"
    assert powershell.policy_version == "labeling/r8-10-v2"
    assert powershell.negative_predicates[0].operator.value == "NOT_ERROR_SIGNATURE"


def test_external_trace_envelope_ref_is_audit_time_independent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    _write_snapshot(
        source,
        content=[{"type": "text", "text": "stable envelope"}],
    )
    index = _index(source, tmp_path / "index")

    first = external_trace_envelope_ref(index, audit=_audit())
    second = external_trace_envelope_ref(
        index,
        audit=_audit(datetime(2026, 8, 10, tzinfo=UTC)),
    )

    assert second == first


def test_structured_reference_matches_search_and_powershell_error(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    _write_snapshot(
        source,
        content=[
            {
                "type": "tool_use",
                "id": "search-1",
                "name": "WebSearch",
                "input": {"query": "evidence"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "search-1",
                "content": "result",
            },
            {
                "type": "tool_use",
                "id": "powershell-1",
                "name": "PowerShell",
                "input": {"command": "Get-Item missing"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "powershell-1",
                "is_error": True,
                "content": "structured failure",
            },
        ],
    )
    index = _index(source, tmp_path / "index")
    specs = approved_external_label_specs(audit=_audit())

    records = _builder().author(
        trace_index=index,
        label_specs=specs,
        audit=_audit(),
    )
    by_name = {record.label_name: record for record in records}

    assert set(by_name) == {
        "search_tool_usage",
        "powershell_error_signature",
    }
    assert all(record.expected_decision is LabelDecisionValueV2.MATCH for record in by_name.values())
    assert all(
        record.reference_state is ExternalReferenceStateV1.REFERENCE_READY for record in by_name.values()
    )
    assert all(
        record.author_kind is ExternalReferenceAuthorKindV1.DETERMINISTIC_SERVICE
        for record in by_name.values()
    )
    assert all(record.evidence_refs for record in by_name.values())
    candidate = by_name["search_tool_usage"].to_r8_candidate()
    assert candidate.expected_decision is LabelDecisionValueV2.MATCH
    assert candidate.annotation_ref == by_name["search_tool_usage"].annotation_ref()


def test_structured_reference_uses_complete_negative_and_incomplete_abstain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    _write_snapshot(source, content=[{"type": "text", "text": "no tools"}])
    index = _index(source, tmp_path / "index")
    specs = approved_external_label_specs(audit=_audit())

    negatives = _builder().author(
        trace_index=index,
        label_specs=specs,
        audit=_audit(),
    )
    assert all(record.expected_decision is LabelDecisionValueV2.NO_MATCH for record in negatives)
    assert all(record.structured_capability_complete for record in negatives)

    recovery = index.normalization_result.recovery_result
    capabilities = tuple(
        TraceCapability(
            capability=capability.capability,
            status=(
                CapabilityStatus.PARTIAL if capability.capability == "tool_events" else capability.status
            ),
        )
        for capability in recovery.capabilities
    )
    partial_recovery = replace(
        recovery,
        parse_quality=ParseQuality.PARTIAL,
        capabilities=capabilities,
    )
    partial_normalization = replace(
        index.normalization_result,
        recovery_result=partial_recovery,
    )
    partial_index = replace(
        index,
        normalization_result=partial_normalization,
    )
    abstained = _builder().author(
        trace_index=partial_index,
        label_specs=specs,
        audit=_audit(),
    )
    search = next(record for record in abstained if record.label_name == "search_tool_usage")
    assert search.expected_decision is LabelDecisionValueV2.ABSTAIN
    assert search.reference_state is ExternalReferenceStateV1.ABSTAINED
    assert search.structured_capability_complete is False


def test_reference_builder_has_no_observation_input() -> None:
    parameters = inspect.signature(ExternalStructuredReferenceBuilder.author).parameters
    assert "observations" not in parameters
    assert "label_decisions" not in parameters
    assert "observation_store" not in parameters


def test_blind_observer_matches_structured_truth_without_reference_access(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    _write_snapshot(
        source,
        content=[
            {
                "type": "tool_use",
                "id": "search-1",
                "name": "WebSearch",
                "input": {"query": "evidence"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "search-1",
                "content": "result",
            },
            {
                "type": "tool_use",
                "id": "powershell-1",
                "name": "PowerShell",
                "input": {"command": "Get-Item missing"},
            },
            {
                "type": "tool_result",
                "tool_use_id": "powershell-1",
                "is_error": True,
                "content": "structured failure",
            },
        ],
    )
    index = _index(source, tmp_path / "index")
    specs = approved_external_label_specs(audit=_audit())
    references = _builder().author(
        trace_index=index,
        label_specs=specs,
        audit=_audit(),
    )
    observation_policy_ref = _ref(
        "external-observation-policy",
        "structured",
        version="v2",
    )
    observer = ExternalBlindStructuredObserver(
        observation_policy_ref=observation_policy_ref,
    )

    observations = observer.observe(
        trace_index=index,
        label_specs=specs,
        audit=_audit(),
    )
    expected = {record.label_spec_ref: record.expected_decision for record in references}
    observed = {record.label_spec_ref: record.label_decision.decision for record in observations}
    search_ref = next(
        record.label_spec_ref for record in references if record.label_name == "search_tool_usage"
    )
    powershell_ref = next(
        record.label_spec_ref for record in references if record.label_name == "powershell_error_signature"
    )

    assert observed[search_ref] is expected[search_ref]
    assert expected[powershell_ref] is LabelDecisionValueV2.MATCH
    assert observed[powershell_ref] is expected[powershell_ref]
    assert all(record.reference_access_denied for record in observations)
    assert all(
        all(not ref.object_type.startswith("external-reference") for ref in record.audit.input_refs)
        for record in observations
    )
    label_refs = tuple(record.label_spec_ref for record in observations)
    source_population_ref = _ref(
        "external-source-population",
        "population",
        version="v2",
    )
    observation_set = BlindLabelObservationSetV1.create(
        source_population_ref=source_population_ref,
        label_spec_refs=label_refs,
        observation_policy_ref=observation_policy_ref,
        records=observations,
        audit=_audit(),
    )
    assert observation_set.reference_access_denied is True
    assert len(observation_set.records) == 2


def test_blind_observer_has_no_reference_input() -> None:
    parameters = inspect.signature(ExternalBlindStructuredObserver.observe).parameters
    assert "references" not in parameters
    assert "reference_store" not in parameters
    assert "expected_decisions" not in parameters
