from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import ClassVar, Self

from pydantic import BaseModel, model_validator

from eval_factory.contracts.core import (
    ContractAudit,
    Identifier,
    ObjectRef,
    Sha256,
)
from eval_factory.contracts.core_v2 import ContractModelV2, canonical_value_v2


class HarnessContractError(ValueError):
    """Raised when declarative Harness authority cannot be composed safely."""


class HarnessObjectV1(ContractModelV2):
    """Content-addressed base for additive Harness contract authorities."""

    object_id: Identifier
    object_sha256: Sha256
    audit: ContractAudit

    OBJECT_TYPE: ClassVar[str]
    OBJECT_VERSION: ClassVar[str] = "v1"

    @classmethod
    def create(cls, *, audit: ContractAudit, **values: object) -> Self:
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=audit,
            **values,  # type: ignore[arg-type]
        )
        refs = collect_object_refs(provisional)
        normalized_audit = audit_with_refs(audit, refs)
        provisional = cls.model_construct(
            object_id=f"{cls.OBJECT_TYPE}://pending",
            object_sha256="0" * 64,
            audit=normalized_audit,
            **values,  # type: ignore[arg-type]
        )
        digest = carried_sha256(provisional)
        return cls(
            object_id=f"{cls.OBJECT_TYPE}://sha256/{digest}",
            object_sha256=digest,
            audit=normalized_audit,
            **values,
        )

    @model_validator(mode="after")
    def validate_derived_identity(self) -> Self:
        expected_sha256 = carried_sha256(self)
        expected_id = f"{self.OBJECT_TYPE}://sha256/{expected_sha256}"
        if self.object_sha256 != expected_sha256 or self.object_id != expected_id:
            raise ValueError("Harness object fields do not match the derived identity")
        if self.audit.input_refs != collect_object_refs(self):
            raise ValueError("Harness object audit refs do not match referenced authority")
        return self

    def to_ref(self) -> ObjectRef:
        return ObjectRef(
            object_type=self.OBJECT_TYPE,
            object_id=self.object_id,
            object_version=self.OBJECT_VERSION,
            object_sha256=self.object_sha256,
        )


def carried_sha256(value: HarnessObjectV1) -> str:
    payload = value.model_dump(
        mode="python",
        exclude={"schema_version", "object_id", "object_sha256", "audit"},
    )
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def collect_object_refs(value: BaseModel) -> tuple[ObjectRef, ...]:
    refs: dict[tuple[str, str, str, str], ObjectRef] = {}

    def visit(item: object) -> None:
        if isinstance(item, ObjectRef):
            refs[ref_key(item)] = item
            return
        if isinstance(item, BaseModel):
            for field_name in type(item).model_fields:
                if field_name in {"audit", "object_id", "object_sha256"}:
                    continue
                visit(getattr(item, field_name))
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, tuple | list | set | frozenset):
            for child in item:
                visit(child)

    visit(value)
    return tuple(refs[key] for key in sorted(refs))


def audit_with_refs(
    audit: ContractAudit,
    refs: tuple[ObjectRef, ...],
) -> ContractAudit:
    return ContractAudit(
        created_at=audit.created_at,
        created_by=audit.created_by,
        governing_versions=audit.governing_versions,
        input_refs=refs,
    )


def ref_key(value: ObjectRef) -> tuple[str, str, str, str]:
    return (
        value.object_type,
        value.object_id,
        value.object_version,
        value.object_sha256,
    )


def sorted_refs(values: Iterable[ObjectRef]) -> tuple[ObjectRef, ...]:
    return tuple(sorted(values, key=ref_key))


def require_sorted_unique(values: tuple[str, ...], label: str) -> None:
    if values != tuple(sorted(values)) or len(values) != len(set(values)):
        raise ValueError(f"{label} must be sorted and unique")


def require_sorted_unique_refs(
    values: tuple[ObjectRef, ...],
    label: str,
) -> None:
    keys = tuple(ref_key(value) for value in values)
    if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
        raise ValueError(f"{label} must be sorted and unique")


def require_ref(
    value: ObjectRef,
    object_type: str,
    label: str,
    *,
    object_version: str = "v1",
) -> None:
    if value.object_type != object_type or value.object_version != object_version:
        raise ValueError(
            f"{label} must reference {object_type}/{object_version}",
        )


def static_object_ref(
    *,
    object_type: str,
    object_id: str,
    object_version: str,
    payload: object,
) -> ObjectRef:
    encoded = json.dumps(
        canonical_value_v2(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return ObjectRef(
        object_type=object_type,
        object_id=object_id,
        object_version=object_version,
        object_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def schema_ref_for(
    model_type: type[BaseModel],
    *,
    schema_id: str,
    object_version: str,
) -> ObjectRef:
    return static_object_ref(
        object_type="json-schema",
        object_id=f"json-schema://{schema_id}",
        object_version=object_version,
        payload=model_type.model_json_schema(),
    )


__all__ = [
    "HarnessContractError",
    "HarnessObjectV1",
    "audit_with_refs",
    "carried_sha256",
    "collect_object_refs",
    "ref_key",
    "require_ref",
    "require_sorted_unique",
    "require_sorted_unique_refs",
    "schema_ref_for",
    "sorted_refs",
    "static_object_ref",
]
