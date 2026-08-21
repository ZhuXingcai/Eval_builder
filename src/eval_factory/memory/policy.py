from __future__ import annotations

import re

from eval_factory.contracts.core import ObjectRef
from eval_factory.memory.models import (
    MemoryAccessContextV1,
    MemoryCandidateV1,
    MemoryNamespaceV1,
    MemorySensitivityV1,
)


class MemoryPolicyError(RuntimeError):
    pass


class MemoryAuthorizationError(MemoryPolicyError):
    pass


class MemoryCapabilityUnavailableError(MemoryPolicyError):
    pass


_SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "PRIVATE_KEY",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "BEARER_TOKEN",
        re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{20,}"),
    ),
    (
        "ASSIGNED_SECRET",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|access[_-]?token)\b"
            r"\s*[:=]\s*[\"']?[a-z0-9._~+/=-]{16,}"
        ),
    ),
)

_FORBIDDEN_SOURCE_TYPES = frozenset(
    {
        "final-output",
        "grader-prompt",
        "grader-rule",
        "hidden-condition",
        "private-reference",
        "reference-answer",
    }
)


class MemoryAdmissionPolicy:
    @staticmethod
    def validate(
        candidate: MemoryCandidateV1,
        *,
        access: MemoryAccessContextV1,
        trusted_approval_refs: frozenset[ObjectRef],
    ) -> None:
        authorize_namespace(candidate.namespace, access=access)
        if candidate.sensitivity is MemorySensitivityV1.FORBIDDEN:
            raise MemoryPolicyError("forbidden memory cannot be persisted")
        if (
            candidate.sensitivity is MemorySensitivityV1.RESTRICTED
            and candidate.approval_ref not in trusted_approval_refs
        ):
            raise MemoryAuthorizationError("memory admission approval is not trusted")
        if any(ref.object_type in _FORBIDDEN_SOURCE_TYPES for ref in candidate.source_refs):
            raise MemoryPolicyError("memory source type is forbidden")
        if _detected_secret_rule(candidate.content) is not None:
            raise MemoryPolicyError("memory content failed the credential policy")


def authorize_namespace(
    namespace: MemoryNamespaceV1,
    *,
    access: MemoryAccessContextV1,
) -> None:
    if not access.authorizes(namespace):
        raise MemoryAuthorizationError("memory namespace is not authorized")


def _detected_secret_rule(content: str) -> str | None:
    for rule_id, pattern in _SECRET_RULES:
        if pattern.search(content):
            return rule_id
    return None


__all__ = [
    "MemoryAdmissionPolicy",
    "MemoryAuthorizationError",
    "MemoryCapabilityUnavailableError",
    "MemoryPolicyError",
    "authorize_namespace",
]
