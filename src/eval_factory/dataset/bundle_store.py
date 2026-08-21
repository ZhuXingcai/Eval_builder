from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from env_mock_agent.facade import (
    LHWorkspaceExportFacade,
    LHWorkspaceExportResultV2,
)
from eval_factory.contracts.release import ReleaseChannel
from eval_factory.contracts.release_publication_v2 import (
    NonProductionReleaseManifestV2,
    validate_nonproduction_release_manifest_v2_identity,
)
from eval_factory.dataset.export import (
    ReleaseBundleFacts,
    ReleasePublicationItemCompilation,
    ReleasePublicationManifestCompilation,
)
from eval_factory.dataset.release_bundle_storage import (
    ImmutableReleaseBundleStorage,
    ReleaseBundleItemPlan,
    manifest_bytes_without_audit,
    prepare_bundle_root,
    roots_overlap,
)


class ReleaseBundlePolicyError(ValueError):
    pass


class ReleaseBundleExportError(RuntimeError):
    pass


class ReleaseBundleIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReleaseBundleWriteResult:
    item_id: str
    channel: ReleaseChannel
    bundle_path: Path
    workspace_export_result: LHWorkspaceExportResultV2
    facts: ReleaseBundleFacts
    reused: bool


class NonProductionReleaseBundleStore:
    def __init__(
        self,
        *,
        canary_root: Path,
        internal_review_root: Path,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        roots = {
            ReleaseChannel.CANARY: prepare_bundle_root(
                canary_root,
                policy_error=ReleaseBundlePolicyError,
                forbidden_names=frozenset({"prod", "production"}),
            ),
            ReleaseChannel.INTERNAL_REVIEW: prepare_bundle_root(
                internal_review_root,
                policy_error=ReleaseBundlePolicyError,
                forbidden_names=frozenset({"prod", "production"}),
            ),
        }
        if roots_overlap(
            roots[ReleaseChannel.CANARY],
            roots[ReleaseChannel.INTERNAL_REVIEW],
        ):
            raise ReleaseBundlePolicyError(
                "non-production release roots must be distinct and non-overlapping"
            )
        self._roots = roots
        self._storage = {
            channel: ImmutableReleaseBundleStorage(
                root,
                policy_error=ReleaseBundlePolicyError,
                export_error=ReleaseBundleExportError,
                integrity_error=ReleaseBundleIntegrityError,
                fault_injector=fault_injector,
            )
            for channel, root in roots.items()
        }

    def build(
        self,
        manifest: ReleasePublicationManifestCompilation,
        *,
        facade: LHWorkspaceExportFacade,
    ) -> tuple[ReleaseBundleWriteResult, ...]:
        validate_nonproduction_release_manifest_v2_identity(manifest.release_manifest)
        if tuple(value.item_manifest for value in manifest.items) != (
            manifest.release_manifest.item_manifests
        ):
            raise ReleaseBundlePolicyError("bundle compilation differs from release manifest")
        storage = self._storage_for(manifest.release_manifest.channel)
        return tuple(
            self._build_item(
                storage,
                manifest.release_manifest,
                item,
                facade=facade,
            )
            for item in manifest.items
        )

    def verify_item(
        self,
        manifest: NonProductionReleaseManifestV2,
        item: ReleasePublicationItemCompilation,
        facts: ReleaseBundleFacts,
    ) -> Path:
        validate_nonproduction_release_manifest_v2_identity(manifest)
        plan = _plan(manifest, item)
        return self._storage_for(manifest.channel).verify_plan(plan, facts)

    def verify_facts(
        self,
        channel: ReleaseChannel,
        facts: ReleaseBundleFacts,
        workspace_result: LHWorkspaceExportResultV2,
    ) -> Path:
        return self._storage_for(channel).verify_facts(
            facts,
            workspace_result,
        )

    def _build_item(
        self,
        storage: ImmutableReleaseBundleStorage,
        manifest: NonProductionReleaseManifestV2,
        item: ReleasePublicationItemCompilation,
        *,
        facade: LHWorkspaceExportFacade,
    ) -> ReleaseBundleWriteResult:
        written = storage.build(_plan(manifest, item), facade=facade)
        return ReleaseBundleWriteResult(
            item_id=written.item_id,
            channel=manifest.channel,
            bundle_path=written.bundle_path,
            workspace_export_result=written.workspace_export_result,
            facts=written.facts,
            reused=written.reused,
        )

    def _storage_for(
        self,
        channel: ReleaseChannel,
    ) -> ImmutableReleaseBundleStorage:
        storage = self._storage.get(channel)
        if storage is None:
            raise ReleaseBundlePolicyError("R7 bundle store accepts only non-production channels")
        return storage


def _plan(
    manifest: NonProductionReleaseManifestV2,
    item: ReleasePublicationItemCompilation,
) -> ReleaseBundleItemPlan:
    if (
        item.item_manifest not in manifest.item_manifests
        or item.item_manifest.channel is not manifest.channel
        or item.item_manifest.registry != manifest.registry
        or item.workspace_request.members != item.item_manifest.expected_workspace_members
        or _sha256(item.query_yaml_bytes) != item.item_manifest.query_yaml_sha256
        or _sha256(item.rubrics_json_bytes) != item.item_manifest.rubrics_json_sha256
    ):
        raise ReleaseBundlePolicyError("bundle Item differs from the release manifest")
    return ReleaseBundleItemPlan(
        item_id=item.item_manifest.item_id,
        query_yaml_bytes=item.query_yaml_bytes,
        rubrics_json_bytes=item.rubrics_json_bytes,
        manifest_bytes=_manifest_bytes_from_value(manifest),
        workspace_request=item.workspace_request,
    )


def _manifest_bytes_from_value(
    manifest: NonProductionReleaseManifestV2,
) -> bytes:
    return manifest_bytes_without_audit(manifest)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "NonProductionReleaseBundleStore",
    "ReleaseBundleExportError",
    "ReleaseBundleIntegrityError",
    "ReleaseBundlePolicyError",
    "ReleaseBundleWriteResult",
]
