from __future__ import annotations

from dataclasses import dataclass

import pytest
from test_support.team_runtime_fixtures import audit, ref

from eval_factory.packs.generic_agent_trace.capabilities import (
    GenericCapabilityAdapterError,
)
from eval_factory.packs.generic_agent_trace.capability_contracts import (
    RequirementPlanningCapabilityRequestV1,
)
from eval_factory.packs.generic_agent_trace.material_resolver import (
    CompositeCapabilityMaterialResolver,
)


@dataclass(frozen=True)
class _Source:
    reference: object
    value: object

    def load(self, reference, expected_type):
        del expected_type
        return self.value if reference == self.reference else None

    def load_many(self, reference, expected_type):
        del expected_type
        return (self.value,) if reference == self.reference else None


def test_material_resolver_validates_owner_refs_and_collections() -> None:
    request = RequirementPlanningCapabilityRequestV1.create(
        plan_ref=ref("dataset-build-plan", "resolver", version="v2"),
        policy_ref=ref("factory-run-policy", "resolver", version="v2"),
        audit=audit(),
    )
    collection_ref = ref(
        "capability-material-collection",
        "resolver",
        version="v2",
    )
    resolver = CompositeCapabilityMaterialResolver()
    resolver.register(request.to_ref(), request)
    resolver.register_many(collection_ref, (request,))
    empty_ref = ref(
        "capability-material-collection",
        "empty",
        version="v2",
    )
    resolver.register_many(empty_ref, ())

    assert (
        resolver.get(
            request.to_ref(),
            RequirementPlanningCapabilityRequestV1,
        )
        == request
    )
    assert resolver.get_many(
        collection_ref,
        RequirementPlanningCapabilityRequestV1,
    ) == (request,)
    assert (
        resolver.get_many(
            empty_ref,
            RequirementPlanningCapabilityRequestV1,
        )
        == ()
    )
    with pytest.raises(GenericCapabilityAdapterError, match="another value"):
        resolver.register(request.to_ref(), object())
    with pytest.raises(GenericCapabilityAdapterError, match="wrong owner type"):
        resolver.get(request.to_ref(), str)


def test_material_resolver_uses_rebuildable_sources_and_fails_closed() -> None:
    request = RequirementPlanningCapabilityRequestV1.create(
        plan_ref=ref("dataset-build-plan", "source", version="v2"),
        policy_ref=ref("factory-run-policy", "source", version="v2"),
        audit=audit(),
    )
    resolver = CompositeCapabilityMaterialResolver(
        (_Source(request.to_ref(), request),),  # type: ignore[arg-type]
    )

    assert (
        resolver.get(
            request.to_ref(),
            RequirementPlanningCapabilityRequestV1,
        )
        == request
    )
    assert resolver.get_many(
        request.to_ref(),
        RequirementPlanningCapabilityRequestV1,
    ) == (request,)
    with pytest.raises(GenericCapabilityAdapterError, match="unavailable"):
        resolver.get(
            ref("missing-material", "resolver", version="v2"),
            RequirementPlanningCapabilityRequestV1,
        )
