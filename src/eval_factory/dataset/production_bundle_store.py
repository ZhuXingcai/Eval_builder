from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from env_mock_agent.facade import (
    LHWorkspaceExportFacade,
    LHWorkspaceExportResultV2,
)
from eval_factory.contracts.production_release_v2 import (
    validate_production_release_manifest_v2_identity,
)
from eval_factory.dataset.export import ReleaseBundleFacts
from eval_factory.dataset.production_release import (
    ProductionReleaseItemCompilation,
    ProductionReleaseManifestCompilation,
)
from eval_factory.dataset.release_bundle_storage import (
    ImmutableReleaseBundleStorage,
    ReleaseBundleItemPlan,
    manifest_bytes_without_audit,
    prepare_bundle_root,
)


class ProductionReleaseBundlePolicyError(ValueError):
    pass


class ProductionReleaseBundleExportError(RuntimeError):
    pass


class ProductionReleaseBundleIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProductionReleaseBundleWriteResult:
    item_id: str
    bundle_path: Path
    workspace_export_result: LHWorkspaceExportResultV2
    facts: ReleaseBundleFacts
    reused: bool


class ProductionReleaseBundleStore:
    def __init__(
        self,
        root: Path,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        candidate = root.expanduser()
        if "production" not in {part.casefold() for part in candidate.parts}:
            raise ProductionReleaseBundlePolicyError(
                "production release root requires an explicit production namespace"
            )
        self.root = prepare_bundle_root(
            candidate,
            policy_error=ProductionReleaseBundlePolicyError,
        )
        self._storage = ImmutableReleaseBundleStorage(
            self.root,
            policy_error=ProductionReleaseBundlePolicyError,
            export_error=ProductionReleaseBundleExportError,
            integrity_error=ProductionReleaseBundleIntegrityError,
            fault_injector=fault_injector,
        )

    def build(
        self,
        manifest: ProductionReleaseManifestCompilation,
        *,
        facade: LHWorkspaceExportFacade,
    ) -> tuple[ProductionReleaseBundleWriteResult, ...]:
        validate_production_release_manifest_v2_identity(manifest.release_manifest)
        if tuple(value.item_manifest for value in manifest.items) != (
            manifest.release_manifest.item_manifests
        ):
            raise ProductionReleaseBundlePolicyError("production bundle compilation differs from manifest")
        return tuple(self._build_item(manifest, item, facade=facade) for item in manifest.items)

    def verify_item(
        self,
        manifest: ProductionReleaseManifestCompilation,
        item: ProductionReleaseItemCompilation,
        facts: ReleaseBundleFacts,
    ) -> Path:
        validate_production_release_manifest_v2_identity(manifest.release_manifest)
        return self._storage.verify_plan(
            _plan(manifest, item),
            facts,
        )

    def verify_facts(
        self,
        facts: ReleaseBundleFacts,
        workspace_result: LHWorkspaceExportResultV2,
    ) -> Path:
        return self._storage.verify_facts(facts, workspace_result)

    def _build_item(
        self,
        manifest: ProductionReleaseManifestCompilation,
        item: ProductionReleaseItemCompilation,
        *,
        facade: LHWorkspaceExportFacade,
    ) -> ProductionReleaseBundleWriteResult:
        written = self._storage.build(
            _plan(manifest, item),
            facade=facade,
        )
        return ProductionReleaseBundleWriteResult(
            item_id=written.item_id,
            bundle_path=written.bundle_path,
            workspace_export_result=written.workspace_export_result,
            facts=written.facts,
            reused=written.reused,
        )


def _plan(
    manifest: ProductionReleaseManifestCompilation,
    item: ProductionReleaseItemCompilation,
) -> ReleaseBundleItemPlan:
    value = item.item_manifest
    if (
        value not in manifest.release_manifest.item_manifests
        or value.registry != manifest.release_manifest.registry
        or item.workspace_request.members != value.expected_workspace_members
        or _sha256(item.query_yaml_bytes) != value.query_yaml_sha256
        or _sha256(item.rubrics_json_bytes) != value.rubrics_json_sha256
    ):
        raise ProductionReleaseBundlePolicyError("production bundle Item differs from manifest")
    return ReleaseBundleItemPlan(
        item_id=value.item_id,
        query_yaml_bytes=item.query_yaml_bytes,
        rubrics_json_bytes=item.rubrics_json_bytes,
        manifest_bytes=manifest_bytes_without_audit(manifest.release_manifest),
        workspace_request=item.workspace_request,
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "ProductionReleaseBundleExportError",
    "ProductionReleaseBundleIntegrityError",
    "ProductionReleaseBundlePolicyError",
    "ProductionReleaseBundleStore",
    "ProductionReleaseBundleWriteResult",
]
