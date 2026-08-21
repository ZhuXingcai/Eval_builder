from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from eval_factory.agent_system.attachment_registry import (
    AttachmentAgentRegistryConfig,
    build_attachment_agent_registry,
)
from eval_factory.contracts.core import ContractAudit, ObjectRef, VersionBinding

HASH = "a" * 64


def _ref(object_type: str, suffix: str) -> ObjectRef:
    return ObjectRef(
        object_type=object_type,
        object_id=f"{object_type}://{suffix}",
        object_version="v2",
        object_sha256=HASH,
    )


def _audit() -> ContractAudit:
    return ContractAudit(
        created_at=datetime(2026, 8, 6, tzinfo=UTC),
        created_by="attachment-registry-test",
        governing_versions=(
            VersionBinding(
                component="graph-engineered-eval-factory",
                version="attachment-registry-v1",
                sha256=HASH,
            ),
        ),
    )


def _config() -> AttachmentAgentRegistryConfig:
    return AttachmentAgentRegistryConfig(
        mock_prompt_ref=_ref("prompt-template", "attachment-mock"),
        quality_prompt_ref=_ref("prompt-template", "attachment-quality"),
        solvability_prompt_ref=_ref(
            "prompt-template",
            "attachment-solvability",
        ),
        mock_model_policy_ref=_ref(
            "model-routing-policy",
            "attachment-mock",
        ),
        quality_model_policy_ref=_ref(
            "model-routing-policy",
            "attachment-quality",
        ),
        solvability_model_policy_ref=_ref(
            "model-routing-policy",
            "attachment-solvability",
        ),
    )


def test_attachment_registry_builds_exact_least_privilege_definitions() -> None:
    registry = build_attachment_agent_registry(
        config=_config(),
        audit=_audit(),
    )

    assert len(registry.capabilities) == 3
    assert len(registry.definitions) == 3

    mock = registry.resolve(
        "attachment-mock-agent",
        "attachment-mock",
    )
    quality = registry.resolve(
        "attachment-quality-agent",
        "attachment-quality",
    )
    solvability = registry.resolve(
        "attachment-solvability-agent",
        "attachment-solvability",
    )

    assert mock.tool_ids == ("attachment-execution",)
    assert mock.data_purpose == "attachment-production"
    assert mock.max_model_requests == 0
    assert mock.max_model_tokens == 0
    assert mock.max_cost_micro_usd == 0
    assert mock.network_allowed is False
    assert mock.workspace_isolated is True

    assert quality.tool_ids == ("item-quality-read",)
    assert quality.data_purpose == "attachment-quality"
    assert quality.max_model_requests == 0
    assert quality.max_model_tokens == 0
    assert quality.max_cost_micro_usd == 0

    assert solvability.tool_ids == ("solvability-evidence-read",)
    assert solvability.data_purpose == "attachment-solvability"
    assert solvability.max_model_requests == 1
    assert solvability.network_allowed is False


def test_attachment_registry_capabilities_do_not_share_write_ownership() -> None:
    registry = build_attachment_agent_registry(
        config=_config(),
        audit=_audit(),
    )

    observed = {
        (
            capability.task_kinds,
            capability.input_object_types,
            capability.output_object_types,
            capability.tool_ids,
            capability.model_capabilities,
        )
        for capability in registry.capabilities
    }

    assert observed == {
        (
            ("attachment-mock",),
            ("attachment-planning-context",),
            ("attachment-group-result",),
            ("attachment-execution",),
            ("structured-output",),
        ),
        (
            ("attachment-quality",),
            (
                "attachment-subgraph-result",
                "item-quality-compilation-result",
            ),
            ("attachment-quality-assessment",),
            ("item-quality-read",),
            (),
        ),
        (
            ("attachment-solvability",),
            (
                "attachment-quality-assessment",
                "solvability-safe-view",
            ),
            ("solvability-assessment",),
            ("solvability-evidence-read",),
            ("reasoning", "structured-output"),
        ),
    }


def test_attachment_registry_rejects_invalid_prompt_authority() -> None:
    config = _config()
    invalid = AttachmentAgentRegistryConfig(
        mock_prompt_ref=_ref("private-prompt-body", "attachment-mock"),
        quality_prompt_ref=config.quality_prompt_ref,
        solvability_prompt_ref=config.solvability_prompt_ref,
        mock_model_policy_ref=config.mock_model_policy_ref,
        quality_model_policy_ref=config.quality_model_policy_ref,
        solvability_model_policy_ref=config.solvability_model_policy_ref,
    )

    with pytest.raises(
        ValidationError,
        match="sensitive object classes",
    ):
        build_attachment_agent_registry(
            config=invalid,
            audit=_audit(),
        )
