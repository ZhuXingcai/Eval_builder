from __future__ import annotations

import hashlib
import importlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    EvidenceRef,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.labeling_v2 import LabelDecisionV2, label_decision_ref

LABEL_BATCH_POLICY_VERSION: Literal["label-batch/r3-05-v1"] = "label-batch/r3-05-v1"


class LabelBatchPolicyError(RuntimeError):
    pass


class LabelPairStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class LabelBatchExportFormat(StrEnum):
    JSONL = "JSONL"
    PARQUET = "PARQUET"


_TERMINAL_PAIR_STATUSES = frozenset({LabelPairStatus.SUCCEEDED, LabelPairStatus.SKIPPED})
_DENIED_EXPORT_OBJECT_TYPES = frozenset(
    {
        "answer-bearing",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "hidden-selection-signal",
        "private-reference",
        "quarantine",
        "quarantine-subject",
        "raw-trace",
        "raw-traj",
        "trace-raw",
    }
)


class LabelBatchPlan(ContractModel):
    schema_version: Literal["eval-factory/label-batch-plan/r3-05"] = "eval-factory/label-batch-plan/r3-05"
    batch_id: Identifier
    trace_envelope_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    label_spec_refs: tuple[ObjectRef, ...] = Field(min_length=1)
    source_dataset_ref: ObjectRef | None = None
    source_job_ref: ObjectRef | None = None
    export_profile: Identifier = "r3-label-analysis"
    policy_version: Literal["label-batch/r3-05-v1"] = LABEL_BATCH_POLICY_VERSION
    plan_sha256: Sha256
    audit: ContractAudit

    @classmethod
    def build(
        cls,
        *,
        trace_envelope_refs: tuple[ObjectRef, ...],
        label_spec_refs: tuple[ObjectRef, ...],
        audit: ContractAudit,
        source_dataset_ref: ObjectRef | None = None,
        source_job_ref: ObjectRef | None = None,
        export_profile: Identifier = "r3-label-analysis",
    ) -> LabelBatchPlan:
        sorted_traces = _sort_refs(trace_envelope_refs)
        sorted_labels = _sort_refs(label_spec_refs)
        seed = {
            "trace_envelope_refs": [_ref_payload(ref) for ref in sorted_traces],
            "label_spec_refs": [_ref_payload(ref) for ref in sorted_labels],
            "source_dataset_ref": None if source_dataset_ref is None else _ref_payload(source_dataset_ref),
            "source_job_ref": None if source_job_ref is None else _ref_payload(source_job_ref),
            "export_profile": export_profile,
            "policy_version": LABEL_BATCH_POLICY_VERSION,
        }
        plan_sha256 = _stable_hash(seed)
        return cls(
            batch_id=_stable_id("label-batch", seed),
            trace_envelope_refs=sorted_traces,
            label_spec_refs=sorted_labels,
            source_dataset_ref=source_dataset_ref,
            source_job_ref=source_job_ref,
            export_profile=export_profile,
            plan_sha256=plan_sha256,
            audit=audit,
        )

    @model_validator(mode="after")
    def validate_plan(self) -> LabelBatchPlan:
        for ref in self.trace_envelope_refs:
            if ref.object_type != "trace-envelope":
                raise ValueError("trace_envelope_refs must reference trace-envelope")
        for ref in self.label_spec_refs:
            if ref.object_type != "label-spec":
                raise ValueError("label_spec_refs must reference label-spec")
        if self.source_dataset_ref is not None and self.source_dataset_ref.object_type != "dataset-job":
            raise ValueError("source_dataset_ref must reference dataset-job")
        if self.source_job_ref is not None and self.source_job_ref.object_type != "dataset-job":
            raise ValueError("source_job_ref must reference dataset-job")
        return self


class LabelBatchPairProgress(ContractModel):
    schema_version: Literal["eval-factory/label-batch-pair-progress/r3-05"] = (
        "eval-factory/label-batch-pair-progress/r3-05"
    )
    pair_id: Identifier
    trace_envelope_ref: ObjectRef
    label_spec_ref: ObjectRef
    status: LabelPairStatus
    attempt_count: int = Field(ge=0)
    label_decision_ref: ObjectRef | None = None
    failure_code: Identifier | None = None
    failure_retryable: bool = False
    updated_reason: str = Field(min_length=1, max_length=512)
    policy_version: Literal["label-batch/r3-05-v1"] = LABEL_BATCH_POLICY_VERSION

    @field_validator("status", mode="before")
    @classmethod
    def parse_status(cls, value: object) -> LabelPairStatus:
        if isinstance(value, LabelPairStatus):
            return value
        if isinstance(value, str):
            return LabelPairStatus(value)
        raise TypeError("status must be a LabelPairStatus")

    @model_validator(mode="after")
    def validate_pair(self) -> LabelBatchPairProgress:
        if self.trace_envelope_ref.object_type != "trace-envelope":
            raise ValueError("pair trace_envelope_ref must reference trace-envelope")
        if self.label_spec_ref.object_type != "label-spec":
            raise ValueError("pair label_spec_ref must reference label-spec")
        if self.label_decision_ref is not None and self.label_decision_ref.object_type != "label-decision":
            raise ValueError("label_decision_ref must reference label-decision")
        if self.status is LabelPairStatus.SUCCEEDED and self.label_decision_ref is None:
            raise ValueError("SUCCEEDED pair requires label_decision_ref")
        if self.status is not LabelPairStatus.SUCCEEDED and self.label_decision_ref is not None:
            raise ValueError("only SUCCEEDED pairs may carry label_decision_ref")
        if self.status is LabelPairStatus.FAILED and self.failure_code is None:
            raise ValueError("FAILED pair requires failure_code")
        return self


class LabelBatchProgressSummary(ContractModel):
    schema_version: Literal["eval-factory/label-batch-progress-summary/r3-05"] = (
        "eval-factory/label-batch-progress-summary/r3-05"
    )
    total: int = Field(ge=0)
    pending: int = Field(ge=0)
    running: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    resumed_completed: int = Field(default=0, ge=0)
    executable: int = Field(default=0, ge=0)


class LabelBatchCheckpoint(ContractModel):
    schema_version: Literal["eval-factory/label-batch-checkpoint/r3-05"] = (
        "eval-factory/label-batch-checkpoint/r3-05"
    )
    checkpoint_id: Identifier
    batch_id: Identifier
    pairs: tuple[LabelBatchPairProgress, ...] = Field(min_length=1)
    summary: LabelBatchProgressSummary
    policy_version: Literal["label-batch/r3-05-v1"] = LABEL_BATCH_POLICY_VERSION
    checkpoint_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_checkpoint(self) -> LabelBatchCheckpoint:
        pair_ids = [pair.pair_id for pair in self.pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("pair IDs must be unique")
        if self.summary.total != len(self.pairs):
            raise ValueError("summary total must match pair count")
        return self


class LabelBatchResumePlan(ContractModel):
    schema_version: Literal["eval-factory/label-batch-resume-plan/r3-05"] = (
        "eval-factory/label-batch-resume-plan/r3-05"
    )
    batch_id: Identifier
    checkpoint_id: Identifier
    executable_pairs: tuple[LabelBatchPairProgress, ...]
    summary: LabelBatchProgressSummary
    policy_version: Literal["label-batch/r3-05-v1"] = LABEL_BATCH_POLICY_VERSION


class LabelBatchExportManifest(ContractModel):
    schema_version: Literal["eval-factory/label-batch-export-manifest/r3-05"] = (
        "eval-factory/label-batch-export-manifest/r3-05"
    )
    manifest_id: Identifier
    batch_id: Identifier
    checkpoint_id: Identifier
    export_format: LabelBatchExportFormat
    output_ref: ObjectRef
    row_count: int = Field(ge=0)
    row_ids: tuple[Identifier, ...]
    content_sha256: Sha256
    policy_version: Literal["label-batch/r3-05-v1"] = LABEL_BATCH_POLICY_VERSION
    manifest_sha256: Sha256
    audit: ContractAudit

    @field_validator("export_format", mode="before")
    @classmethod
    def parse_export_format(cls, value: object) -> LabelBatchExportFormat:
        if isinstance(value, LabelBatchExportFormat):
            return value
        if isinstance(value, str):
            return LabelBatchExportFormat(value)
        raise TypeError("export_format must be a LabelBatchExportFormat")

    @model_validator(mode="after")
    def validate_manifest(self) -> LabelBatchExportManifest:
        if self.output_ref.object_type != "label-batch-export":
            raise ValueError("output_ref must reference label-batch-export")
        if self.row_count != len(self.row_ids):
            raise ValueError("row_count must match row_ids")
        return self


class LabelBatchProgressTracker:
    policy_version = LABEL_BATCH_POLICY_VERSION

    def initialize(self, plan: LabelBatchPlan, *, audit: ContractAudit) -> LabelBatchCheckpoint:
        pairs = tuple(
            LabelBatchPairProgress(
                pair_id=_pair_id(plan.batch_id, trace_ref, label_ref),
                trace_envelope_ref=trace_ref,
                label_spec_ref=label_ref,
                status=LabelPairStatus.PENDING,
                attempt_count=0,
                updated_reason="initialized",
            )
            for trace_ref in plan.trace_envelope_refs
            for label_ref in plan.label_spec_refs
        )
        return _checkpoint(batch_id=plan.batch_id, pairs=_sort_pairs(pairs), audit=audit)

    def record_started(
        self,
        checkpoint: LabelBatchCheckpoint,
        *,
        pair_id: Identifier,
        audit: ContractAudit,
    ) -> LabelBatchCheckpoint:
        pair = _get_pair(checkpoint, pair_id)
        _ensure_not_terminal(pair)
        return _replace_pair(
            checkpoint,
            LabelBatchPairProgress(
                pair_id=pair.pair_id,
                trace_envelope_ref=pair.trace_envelope_ref,
                label_spec_ref=pair.label_spec_ref,
                status=LabelPairStatus.RUNNING,
                attempt_count=pair.attempt_count + 1,
                updated_reason="started",
            ),
            audit=audit,
        )

    def record_decision(
        self,
        checkpoint: LabelBatchCheckpoint,
        *,
        pair_id: Identifier,
        decision: LabelDecisionV2,
        audit: ContractAudit,
    ) -> LabelBatchCheckpoint:
        pair = _get_pair(checkpoint, pair_id)
        _ensure_not_terminal(pair)
        _validate_decision_for_pair(pair, decision)
        return _replace_pair(
            checkpoint,
            LabelBatchPairProgress(
                pair_id=pair.pair_id,
                trace_envelope_ref=pair.trace_envelope_ref,
                label_spec_ref=pair.label_spec_ref,
                status=LabelPairStatus.SUCCEEDED,
                attempt_count=max(pair.attempt_count, 1),
                label_decision_ref=self.decision_ref(decision),
                updated_reason="decision-recorded",
            ),
            audit=audit,
        )

    def record_failure(
        self,
        checkpoint: LabelBatchCheckpoint,
        *,
        pair_id: Identifier,
        failure_code: Identifier,
        retryable: bool,
        audit: ContractAudit,
    ) -> LabelBatchCheckpoint:
        pair = _get_pair(checkpoint, pair_id)
        _ensure_not_terminal(pair)
        return _replace_pair(
            checkpoint,
            LabelBatchPairProgress(
                pair_id=pair.pair_id,
                trace_envelope_ref=pair.trace_envelope_ref,
                label_spec_ref=pair.label_spec_ref,
                status=LabelPairStatus.FAILED,
                attempt_count=max(pair.attempt_count, 1),
                failure_code=failure_code,
                failure_retryable=retryable,
                updated_reason="failure-recorded",
            ),
            audit=audit,
        )

    def record_skipped(
        self,
        checkpoint: LabelBatchCheckpoint,
        *,
        pair_id: Identifier,
        reason: str,
        audit: ContractAudit,
    ) -> LabelBatchCheckpoint:
        pair = _get_pair(checkpoint, pair_id)
        _ensure_not_terminal(pair)
        return _replace_pair(
            checkpoint,
            LabelBatchPairProgress(
                pair_id=pair.pair_id,
                trace_envelope_ref=pair.trace_envelope_ref,
                label_spec_ref=pair.label_spec_ref,
                status=LabelPairStatus.SKIPPED,
                attempt_count=pair.attempt_count,
                updated_reason=reason,
            ),
            audit=audit,
        )

    def resume_work(self, checkpoint: LabelBatchCheckpoint) -> LabelBatchResumePlan:
        executable = tuple(pair for pair in checkpoint.pairs if _is_executable(pair))
        summary = _summary(
            checkpoint.pairs,
            resumed_completed=sum(1 for pair in checkpoint.pairs if pair.status in _TERMINAL_PAIR_STATUSES),
            executable=len(executable),
        )
        return LabelBatchResumePlan(
            batch_id=checkpoint.batch_id,
            checkpoint_id=checkpoint.checkpoint_id,
            executable_pairs=executable,
            summary=summary,
        )

    @staticmethod
    def decision_ref(decision: LabelDecisionV2) -> ObjectRef:
        return label_decision_ref(decision)


class LabelBatchExporter:
    policy_version = LABEL_BATCH_POLICY_VERSION

    def write_jsonl(
        self,
        checkpoint: LabelBatchCheckpoint,
        *,
        decisions: tuple[LabelDecisionV2, ...],
        path: Path,
        audit: ContractAudit,
    ) -> LabelBatchExportManifest:
        rows = self.rows(checkpoint, decisions=decisions)
        content = _jsonl_bytes(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return _export_manifest(
            checkpoint=checkpoint,
            rows=rows,
            content=content,
            export_format=LabelBatchExportFormat.JSONL,
            audit=audit,
        )

    def write_parquet(
        self,
        checkpoint: LabelBatchCheckpoint,
        *,
        decisions: tuple[LabelDecisionV2, ...],
        path: Path,
        audit: ContractAudit,
    ) -> LabelBatchExportManifest:
        rows = self.rows(checkpoint, decisions=decisions)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd = cast(Any, importlib.import_module("pandas"))
        frame = pd.DataFrame(rows, columns=LABEL_BATCH_EXPORT_ROW_FIELDS)
        frame.to_parquet(path, index=False)
        return _export_manifest(
            checkpoint=checkpoint,
            rows=rows,
            content=path.read_bytes(),
            export_format=LabelBatchExportFormat.PARQUET,
            audit=audit,
        )

    def rows(
        self,
        checkpoint: LabelBatchCheckpoint,
        *,
        decisions: tuple[LabelDecisionV2, ...],
    ) -> tuple[dict[str, object], ...]:
        decision_index = _decision_index(decisions)
        rows = []
        for pair in checkpoint.pairs:
            if pair.status is not LabelPairStatus.SUCCEEDED:
                continue
            if pair.label_decision_ref is None:
                raise LabelBatchPolicyError("succeeded pair is missing label decision ref")
            decision = decision_index.get(_ref_key(pair.label_decision_ref))
            if decision is None:
                raise LabelBatchPolicyError("stale or missing label decision")
            _validate_decision_for_pair(pair, decision)
            rows.append(_export_row(checkpoint, pair, decision, pair.label_decision_ref))
        return tuple(sorted(rows, key=lambda row: str(row["row_id"])))


LABEL_BATCH_EXPORT_ROW_FIELDS = (
    "row_id",
    "batch_id",
    "checkpoint_id",
    "pair_id",
    "trace_envelope_ref_object_type",
    "trace_envelope_ref_object_id",
    "trace_envelope_ref_object_version",
    "trace_envelope_ref_sha256",
    "label_spec_ref_object_type",
    "label_spec_ref_object_id",
    "label_spec_ref_object_version",
    "label_spec_ref_sha256",
    "label_decision_ref_object_type",
    "label_decision_ref_object_id",
    "label_decision_ref_object_version",
    "label_decision_ref_sha256",
    "decision",
    "execution_status",
    "pair_status",
    "structured_capability_complete",
    "confidence",
    "positive_evidence_ref_ids",
    "negative_evidence_ref_ids",
    "semantic_evidence_ref_ids",
    "unresolved_reasons",
    "model_profile",
    "prompt_version",
    "rule_version",
    "label_decision_policy_version",
    "label_decision_sha256",
    "batch_policy_version",
    "checkpoint_sha256",
)


def _checkpoint(
    *,
    batch_id: Identifier,
    pairs: tuple[LabelBatchPairProgress, ...],
    audit: ContractAudit,
) -> LabelBatchCheckpoint:
    sorted_pairs = _sort_pairs(pairs)
    summary = _summary(sorted_pairs)
    seed = _checkpoint_seed(batch_id, sorted_pairs)
    return LabelBatchCheckpoint(
        checkpoint_id=_stable_id("label-batch-checkpoint", seed),
        batch_id=batch_id,
        pairs=sorted_pairs,
        summary=summary,
        checkpoint_sha256=_stable_hash(seed),
        audit=audit,
    )


def _replace_pair(
    checkpoint: LabelBatchCheckpoint,
    replacement: LabelBatchPairProgress,
    *,
    audit: ContractAudit,
) -> LabelBatchCheckpoint:
    pairs = tuple(replacement if pair.pair_id == replacement.pair_id else pair for pair in checkpoint.pairs)
    return _checkpoint(batch_id=checkpoint.batch_id, pairs=pairs, audit=audit)


def _summary(
    pairs: tuple[LabelBatchPairProgress, ...],
    *,
    resumed_completed: int = 0,
    executable: int = 0,
) -> LabelBatchProgressSummary:
    return LabelBatchProgressSummary(
        total=len(pairs),
        pending=sum(1 for pair in pairs if pair.status is LabelPairStatus.PENDING),
        running=sum(1 for pair in pairs if pair.status is LabelPairStatus.RUNNING),
        succeeded=sum(1 for pair in pairs if pair.status is LabelPairStatus.SUCCEEDED),
        failed=sum(1 for pair in pairs if pair.status is LabelPairStatus.FAILED),
        skipped=sum(1 for pair in pairs if pair.status is LabelPairStatus.SKIPPED),
        resumed_completed=resumed_completed,
        executable=executable,
    )


def _checkpoint_seed(batch_id: str, pairs: tuple[LabelBatchPairProgress, ...]) -> dict[str, object]:
    return {
        "batch_id": batch_id,
        "pairs": [_pair_payload(pair) for pair in pairs],
        "policy_version": LABEL_BATCH_POLICY_VERSION,
    }


def _pair_payload(pair: LabelBatchPairProgress) -> dict[str, object]:
    return {
        "pair_id": pair.pair_id,
        "trace_envelope_ref": _ref_payload(pair.trace_envelope_ref),
        "label_spec_ref": _ref_payload(pair.label_spec_ref),
        "status": pair.status.value,
        "attempt_count": pair.attempt_count,
        "label_decision_ref": None
        if pair.label_decision_ref is None
        else _ref_payload(pair.label_decision_ref),
        "failure_code": pair.failure_code,
        "failure_retryable": pair.failure_retryable,
        "updated_reason": pair.updated_reason,
        "policy_version": pair.policy_version,
    }


def _pair_id(batch_id: str, trace_ref: ObjectRef, label_ref: ObjectRef) -> str:
    return _stable_id(
        "label-batch-pair",
        {
            "batch_id": batch_id,
            "trace_envelope_ref": _ref_payload(trace_ref),
            "label_spec_ref": _ref_payload(label_ref),
            "policy_version": LABEL_BATCH_POLICY_VERSION,
        },
    )


def _get_pair(checkpoint: LabelBatchCheckpoint, pair_id: str) -> LabelBatchPairProgress:
    for pair in checkpoint.pairs:
        if pair.pair_id == pair_id:
            return pair
    raise LabelBatchPolicyError(f"unknown label batch pair: {pair_id}")


def _ensure_not_terminal(pair: LabelBatchPairProgress) -> None:
    if pair.status in _TERMINAL_PAIR_STATUSES:
        raise LabelBatchPolicyError(f"terminal pair cannot be updated: {pair.pair_id}")


def _is_executable(pair: LabelBatchPairProgress) -> bool:
    if pair.status in {LabelPairStatus.PENDING, LabelPairStatus.RUNNING}:
        return True
    return bool(pair.status is LabelPairStatus.FAILED and pair.failure_retryable)


def _validate_decision_for_pair(pair: LabelBatchPairProgress, decision: LabelDecisionV2) -> None:
    expected_ref = LabelBatchProgressTracker.decision_ref(decision)
    if pair.label_decision_ref is not None and pair.label_decision_ref != expected_ref:
        raise LabelBatchPolicyError("stale or missing label decision")
    if decision.trace_envelope_ref != pair.trace_envelope_ref:
        raise LabelBatchPolicyError("label decision trace envelope ref mismatch")
    if decision.label_spec_ref != pair.label_spec_ref:
        raise LabelBatchPolicyError("label decision label spec ref mismatch")
    for evidence in (
        *decision.positive_evidence,
        *decision.negative_evidence,
        *decision.semantic_evidence,
    ):
        if evidence.subject_ref.object_type in _DENIED_EXPORT_OBJECT_TYPES:
            raise LabelBatchPolicyError("unsafe label decision evidence reference")


def _decision_index(
    decisions: tuple[LabelDecisionV2, ...],
) -> dict[tuple[str, str, str, str], LabelDecisionV2]:
    index: dict[tuple[str, str, str, str], LabelDecisionV2] = {}
    for decision in decisions:
        ref = LabelBatchProgressTracker.decision_ref(decision)
        key = _ref_key(ref)
        if key in index:
            raise LabelBatchPolicyError("duplicate label decision ref")
        index[key] = decision
    return index


def _export_row(
    checkpoint: LabelBatchCheckpoint,
    pair: LabelBatchPairProgress,
    decision: LabelDecisionV2,
    decision_ref: ObjectRef,
) -> dict[str, object]:
    trace = pair.trace_envelope_ref
    label = pair.label_spec_ref
    row_seed = {
        "batch_id": checkpoint.batch_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "pair_id": pair.pair_id,
        "label_decision_ref": _ref_payload(decision_ref),
        "policy_version": LABEL_BATCH_POLICY_VERSION,
    }
    return {
        "row_id": _stable_id("label-batch-row", row_seed),
        "batch_id": checkpoint.batch_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "pair_id": pair.pair_id,
        "trace_envelope_ref_object_type": trace.object_type,
        "trace_envelope_ref_object_id": trace.object_id,
        "trace_envelope_ref_object_version": trace.object_version,
        "trace_envelope_ref_sha256": trace.object_sha256,
        "label_spec_ref_object_type": label.object_type,
        "label_spec_ref_object_id": label.object_id,
        "label_spec_ref_object_version": label.object_version,
        "label_spec_ref_sha256": label.object_sha256,
        "label_decision_ref_object_type": decision_ref.object_type,
        "label_decision_ref_object_id": decision_ref.object_id,
        "label_decision_ref_object_version": decision_ref.object_version,
        "label_decision_ref_sha256": decision_ref.object_sha256,
        "decision": decision.decision.value,
        "execution_status": decision.execution_status.value,
        "pair_status": pair.status.value,
        "structured_capability_complete": decision.structured_capability_complete,
        "confidence": decision.confidence,
        "positive_evidence_ref_ids": _evidence_ids(decision.positive_evidence),
        "negative_evidence_ref_ids": _evidence_ids(decision.negative_evidence),
        "semantic_evidence_ref_ids": _evidence_ids(decision.semantic_evidence),
        "unresolved_reasons": sorted(reason.value for reason in decision.unresolved_reasons),
        "model_profile": decision.model_profile,
        "prompt_version": decision.prompt_version,
        "rule_version": decision.rule_version,
        "label_decision_policy_version": decision.policy_version,
        "label_decision_sha256": decision.decision_sha256,
        "batch_policy_version": LABEL_BATCH_POLICY_VERSION,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }


def _evidence_ids(evidence: tuple[EvidenceRef, ...]) -> list[str]:
    return sorted(item.evidence_ref_id for item in evidence)


def _jsonl_bytes(rows: tuple[dict[str, object], ...]) -> bytes:
    if not rows:
        return b""
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for row in rows
        )
        + "\n"
    ).encode()


def _export_manifest(
    *,
    checkpoint: LabelBatchCheckpoint,
    rows: tuple[dict[str, object], ...],
    content: bytes,
    export_format: LabelBatchExportFormat,
    audit: ContractAudit,
) -> LabelBatchExportManifest:
    content_sha256 = hashlib.sha256(content).hexdigest()
    row_ids = tuple(str(row["row_id"]) for row in rows)
    output_ref = ObjectRef(
        object_type="label-batch-export",
        object_id=_stable_id("label-batch-export", {"content_sha256": content_sha256}),
        object_version=f"{export_format.value}/{LABEL_BATCH_POLICY_VERSION}",
        object_sha256=content_sha256,
    )
    seed = {
        "batch_id": checkpoint.batch_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "export_format": export_format.value,
        "output_ref": _ref_payload(output_ref),
        "row_count": len(rows),
        "row_ids": list(row_ids),
        "content_sha256": content_sha256,
        "policy_version": LABEL_BATCH_POLICY_VERSION,
    }
    return LabelBatchExportManifest(
        manifest_id=_stable_id("label-batch-export-manifest", seed),
        batch_id=checkpoint.batch_id,
        checkpoint_id=checkpoint.checkpoint_id,
        export_format=export_format,
        output_ref=output_ref,
        row_count=len(rows),
        row_ids=row_ids,
        content_sha256=content_sha256,
        manifest_sha256=_stable_hash(seed),
        audit=audit,
    )


def _sort_refs(refs: tuple[ObjectRef, ...]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(refs, key=lambda ref: (ref.object_id, ref.object_version, ref.object_sha256)))


def _sort_pairs(pairs: tuple[LabelBatchPairProgress, ...]) -> tuple[LabelBatchPairProgress, ...]:
    return tuple(sorted(pairs, key=lambda pair: pair.pair_id))


def _ref_payload(ref: ObjectRef) -> dict[str, str]:
    return {
        "object_type": ref.object_type,
        "object_id": ref.object_id,
        "object_version": ref.object_version,
        "object_sha256": ref.object_sha256,
    }


def _ref_key(ref: ObjectRef) -> tuple[str, str, str, str]:
    return (ref.object_type, ref.object_id, ref.object_version, ref.object_sha256)


def _stable_id(kind: str, payload: object) -> str:
    return f"{kind}://sha256/{_stable_hash(payload)}"


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()
