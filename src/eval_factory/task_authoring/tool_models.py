from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import Field, field_validator, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    ContractModel,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import canonical_value_v2
from eval_factory.contracts.task_v2 import (
    ContestantToolPolicyV2,
    ToolPolicyV2,
)
from eval_factory.contracts.trace import ToolFamily

TOOL_POLICY_POLICY_VERSION: Literal["tool-policy/r4-07-v1"] = "tool-policy/r4-07-v1"

_DENIED_STATIC_REF_MARKERS = frozenset(
    {
        "answer-bearing",
        "command",
        "configured-pii",
        "credential",
        "final-answer",
        "final-output",
        "grader-rule",
        "hidden-pass-condition",
        "path-value",
        "private-reference",
        "prompt-text",
        "quarantine",
        "raw-trace",
        "restricted-pii",
        "runtime-credential",
        "secret",
        "sensitive-pii",
        "vendor-tool",
    }
)


class ToolPolicyPolicyError(RuntimeError):
    pass


class ToolPolicyCompilationOutcome(StrEnum):
    COMPILED = "COMPILED"
    BLOCKED_CAPABILITY = "BLOCKED_CAPABILITY"
    BLOCKED_POLICY = "BLOCKED_POLICY"


class ToolPolicyCompilationReason(StrEnum):
    MISSING_CATALOG_DEFINITION = "MISSING_CATALOG_DEFINITION"
    TOOL_NOT_CONTESTANT_ELIGIBLE = "TOOL_NOT_CONTESTANT_ELIGIBLE"
    UNKNOWN_TOOL_FAMILY = "UNKNOWN_TOOL_FAMILY"


class ToolCapabilityDefinition(ContractModel):
    schema_version: Literal["eval-factory/tool-capability-definition/r4-07"] = (
        "eval-factory/tool-capability-definition/r4-07"
    )
    definition_id: Identifier
    tool_id: Identifier
    tool_family: ToolFamily
    capability_ref: ObjectRef
    enforcement_profile_ref: ObjectRef
    constraint_profile_ref: ObjectRef
    contestant_descriptor_ref: ObjectRef
    contestant_constraint_profile_ref: ObjectRef
    contestant_eligible: bool
    definition_sha256: Sha256

    @field_validator("tool_family", mode="before")
    @classmethod
    def parse_tool_family(cls, value: object) -> ToolFamily:
        if isinstance(value, ToolFamily):
            return value
        if isinstance(value, str):
            return ToolFamily(value)
        raise TypeError("tool_family must be a ToolFamily")

    @model_validator(mode="after")
    def validate_definition(self) -> ToolCapabilityDefinition:
        _require_ref_type(self.capability_ref, "tool-capability", "capability_ref")
        _require_ref_type(
            self.enforcement_profile_ref,
            "tool-enforcement-profile",
            "enforcement_profile_ref",
        )
        _require_ref_type(
            self.constraint_profile_ref,
            "tool-constraint-profile",
            "constraint_profile_ref",
        )
        _require_ref_type(
            self.contestant_descriptor_ref,
            "contestant-tool-descriptor",
            "contestant_descriptor_ref",
        )
        _require_ref_type(
            self.contestant_constraint_profile_ref,
            "contestant-tool-constraint-profile",
            "contestant_constraint_profile_ref",
        )
        for ref in (
            self.capability_ref,
            self.enforcement_profile_ref,
            self.constraint_profile_ref,
            self.contestant_descriptor_ref,
            self.contestant_constraint_profile_ref,
        ):
            _require_content_free_static_ref(ref)
        if self.tool_family is ToolFamily.UNKNOWN and self.contestant_eligible:
            raise ValueError("UNKNOWN tool family cannot be contestant eligible")
        return self


class ToolCapabilityCatalog(ContractModel):
    schema_version: Literal["eval-factory/tool-capability-catalog/r4-07"] = (
        "eval-factory/tool-capability-catalog/r4-07"
    )
    catalog_id: Identifier
    catalog_version: str = Field(min_length=1, max_length=128)
    definitions: tuple[ToolCapabilityDefinition, ...] = Field(min_length=1)
    catalog_sha256: Sha256
    audit: ContractAudit

    @model_validator(mode="after")
    def validate_catalog(self) -> ToolCapabilityCatalog:
        _require_unique(
            "definition IDs",
            tuple(item.definition_id for item in self.definitions),
        )
        _require_unique("tool IDs", tuple(item.tool_id for item in self.definitions))
        _require_unique(
            "definition hashes",
            tuple(item.definition_sha256 for item in self.definitions),
        )
        return self


class ToolPolicyCompilationResult(ContractModel):
    schema_version: Literal["eval-factory/tool-policy-compilation-result/r4-07"] = (
        "eval-factory/tool-policy-compilation-result/r4-07"
    )
    result_id: Identifier
    task_draft_ref: ObjectRef
    rubric_set_ref: ObjectRef
    evaluator_spec_ref: ObjectRef
    tool_catalog_ref: ObjectRef
    outcome: ToolPolicyCompilationOutcome
    tool_policy: ToolPolicyV2 | None = None
    contestant_projection: ContestantToolPolicyV2 | None = None
    unresolved_reasons: frozenset[ToolPolicyCompilationReason] = frozenset()
    policy_version: Literal["tool-policy/r4-07-v1"] = TOOL_POLICY_POLICY_VERSION
    result_sha256: Sha256
    audit: ContractAudit

    @field_validator("outcome", mode="before")
    @classmethod
    def parse_outcome(cls, value: object) -> ToolPolicyCompilationOutcome:
        if isinstance(value, ToolPolicyCompilationOutcome):
            return value
        if isinstance(value, str):
            return ToolPolicyCompilationOutcome(value)
        raise TypeError("outcome must be a ToolPolicyCompilationOutcome")

    @field_validator("unresolved_reasons", mode="before")
    @classmethod
    def parse_reasons(
        cls,
        value: object,
    ) -> frozenset[ToolPolicyCompilationReason]:
        if isinstance(value, (frozenset, set, tuple, list)):
            return frozenset(
                item if isinstance(item, ToolPolicyCompilationReason) else ToolPolicyCompilationReason(item)
                for item in value
            )
        raise TypeError("unresolved_reasons must be a collection")

    @model_validator(mode="after")
    def validate_result(self) -> ToolPolicyCompilationResult:
        _require_v2_ref(self.task_draft_ref, "task-draft", "task_draft_ref")
        _require_v2_ref(self.rubric_set_ref, "rubric-set", "rubric_set_ref")
        _require_v2_ref(self.evaluator_spec_ref, "evaluator-spec", "evaluator_spec_ref")
        _require_ref_type(
            self.tool_catalog_ref,
            "tool-capability-catalog",
            "tool_catalog_ref",
        )
        if self.outcome is ToolPolicyCompilationOutcome.COMPILED:
            if self.tool_policy is None or self.contestant_projection is None:
                raise ValueError("COMPILED result requires policy and contestant projection")
            if self.unresolved_reasons:
                raise ValueError("COMPILED result cannot carry unresolved reasons")
        else:
            if self.tool_policy is not None or self.contestant_projection is not None:
                raise ValueError(f"{self.outcome.value} result cannot carry compiled outputs")
            if not self.unresolved_reasons:
                raise ValueError(f"{self.outcome.value} result requires unresolved reasons")
        return self


def tool_capability_definition_carried_sha256(
    definition: ToolCapabilityDefinition,
) -> str:
    return tool_policy_payload_sha256(
        {
            "tool_id": definition.tool_id,
            "tool_family": definition.tool_family.value,
            "capability_ref": _ref_payload(definition.capability_ref),
            "enforcement_profile_ref": _ref_payload(definition.enforcement_profile_ref),
            "constraint_profile_ref": _ref_payload(definition.constraint_profile_ref),
            "contestant_descriptor_ref": _ref_payload(definition.contestant_descriptor_ref),
            "contestant_constraint_profile_ref": _ref_payload(definition.contestant_constraint_profile_ref),
            "contestant_eligible": definition.contestant_eligible,
        }
    )


def tool_capability_catalog_carried_sha256(
    catalog: ToolCapabilityCatalog,
) -> str:
    return tool_policy_payload_sha256(
        {
            "catalog_version": catalog.catalog_version,
            "definitions": [
                item.model_dump(mode="json", exclude_none=False)
                for item in sorted(
                    catalog.definitions,
                    key=lambda definition: definition.tool_id,
                )
            ],
        }
    )


def tool_capability_catalog_ref(catalog: ToolCapabilityCatalog) -> ObjectRef:
    return ObjectRef(
        object_type="tool-capability-catalog",
        object_id=catalog.catalog_id,
        object_version=catalog.catalog_version,
        object_sha256=catalog.catalog_sha256,
    )


def tool_policy_payload_sha256(payload: object) -> str:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_ref_type(ref: ObjectRef, expected: str, field_name: str) -> None:
    if ref.object_type != expected:
        raise ValueError(f"{field_name} must reference {expected}")


def _require_v2_ref(ref: ObjectRef, expected: str, field_name: str) -> None:
    _require_ref_type(ref, expected, field_name)
    if ref.object_version != "v2":
        raise ValueError(f"{field_name} must reference {expected} v2")


def _require_content_free_static_ref(ref: ObjectRef) -> None:
    normalized = f"{ref.object_type}:{ref.object_id}".casefold().replace("_", "-")
    if any(marker in normalized for marker in _DENIED_STATIC_REF_MARKERS):
        raise ValueError("tool capability definition refs must be content-free static identifiers")


def _require_unique(label: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique")


def _ref_payload(ref: ObjectRef) -> dict[str, object]:
    return ref.model_dump(mode="json", exclude_none=False)
