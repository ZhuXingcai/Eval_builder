from __future__ import annotations

from pathlib import Path

import pytest
from test_production_release import (
    ROOT,
    _audit,
    _authority,
    _production_policy,
)
from test_release_publication import _source

from env_mock_agent.facade.release_export_adapter import (
    MappingLHWorkspaceExportMaterialResolver,
    RegistryLHWorkspaceExportFacade,
)
from eval_factory.dataset.production_bundle_store import (
    ProductionReleaseBundleIntegrityError,
    ProductionReleaseBundlePolicyError,
    ProductionReleaseBundleStore,
)
from eval_factory.dataset.production_release import (
    ProductionReleaseCompiler,
    ProductionReleasePredecessor,
)


def _manifest():
    source, _ = _source()
    policy = _production_policy()
    profile = (ROOT / "specs/002-eval-dataset-factory/release-profiles/v1/decision.json").read_bytes()
    return ProductionReleaseCompiler().compile_manifest(
        job_id=source.approved_result.release_subject.job_id,
        item_sources=(source,),
        predecessors=(ProductionReleasePredecessor.direct(source.approved_result),),
        attestation=_authority(),
        release_profile_payload=profile,
        registry="registry://production/lh-v1",
        policy=policy,
        audit=_audit(),
    )


def test_production_bundle_builds_reuses_and_verifies(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    facade = RegistryLHWorkspaceExportFacade(resolver=MappingLHWorkspaceExportMaterialResolver(outputs={}))
    store = ProductionReleaseBundleStore(
        tmp_path / "production/bundles",
    )

    first = store.build(manifest, facade=facade)
    second = store.build(manifest, facade=facade)

    assert len(first) == 1
    assert first[0].reused is False
    assert second[0].reused is True
    assert second[0].facts == first[0].facts
    assert (
        store.verify_facts(
            first[0].facts,
            first[0].workspace_export_result,
        )
        == first[0].bundle_path
    )


def test_production_bundle_rejects_nonproduction_root_and_corruption(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProductionReleaseBundlePolicyError, match="production"):
        ProductionReleaseBundleStore(tmp_path / "canary/bundles")
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)
    with pytest.raises(
        ProductionReleaseBundlePolicyError,
        match="real directory",
    ):
        ProductionReleaseBundleStore(linked / "production/bundles")

    manifest = _manifest()
    facade = RegistryLHWorkspaceExportFacade(resolver=MappingLHWorkspaceExportMaterialResolver(outputs={}))
    store = ProductionReleaseBundleStore(tmp_path / "production/bundles")
    result = store.build(manifest, facade=facade)[0]
    (result.bundle_path / "query.yaml").write_text("changed", encoding="utf-8")

    with pytest.raises(ProductionReleaseBundleIntegrityError):
        store.verify_facts(result.facts, result.workspace_export_result)


def test_production_bundle_rejects_extra_files_and_symlinks(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    facade = RegistryLHWorkspaceExportFacade(resolver=MappingLHWorkspaceExportMaterialResolver(outputs={}))
    extra_store = ProductionReleaseBundleStore(tmp_path / "production/extra")
    extra = extra_store.build(manifest, facade=facade)[0]
    (extra.bundle_path / "workspace/unexpected.txt").write_text(
        "unexpected",
        encoding="utf-8",
    )
    with pytest.raises(
        ProductionReleaseBundleIntegrityError,
        match="facts",
    ):
        extra_store.verify_facts(
            extra.facts,
            extra.workspace_export_result,
        )

    link_store = ProductionReleaseBundleStore(tmp_path / "production/link")
    linked = link_store.build(manifest, facade=facade)[0]
    (linked.bundle_path / "workspace/link").symlink_to(linked.bundle_path / "query.yaml")
    with pytest.raises(
        ProductionReleaseBundleIntegrityError,
        match="symlink",
    ):
        link_store.verify_facts(
            linked.facts,
            linked.workspace_export_result,
        )


def test_production_bundle_after_rename_crash_is_reusable(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    facade = RegistryLHWorkspaceExportFacade(resolver=MappingLHWorkspaceExportMaterialResolver(outputs={}))

    def fault(point: str) -> None:
        if point == "after_rename":
            raise RuntimeError("simulated production bundle crash")

    with pytest.raises(RuntimeError, match="simulated"):
        ProductionReleaseBundleStore(
            tmp_path / "production/bundles",
            fault_injector=fault,
        ).build(manifest, facade=facade)

    replay = ProductionReleaseBundleStore(tmp_path / "production/bundles").build(manifest, facade=facade)[0]

    assert replay.reused is True
