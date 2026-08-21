from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from test_release_publication import (
    _compile_manifest,
    _source,
)

from env_mock_agent.facade import (
    FacadeObjectRef,
    LHWorkspaceExportFailureCodeV2,
    LHWorkspaceExportMemberTypeV2,
    LHWorkspaceExportMemberV2,
    LHWorkspaceExportRequestV2,
    LHWorkspaceExportResultV2,
)
from env_mock_agent.facade.release_export_adapter import (
    MappingLHWorkspaceExportMaterialResolver,
    RegistryLHWorkspaceExportFacade,
)
from env_mock_agent.providers.helpers import sha256_path
from eval_factory.contracts.release_publication_v2 import (
    LHReleaseItemManifestV2,
    NonProductionReleaseManifestV2,
)
from eval_factory.dataset.bundle_store import (
    NonProductionReleaseBundleStore,
    ReleaseBundleExportError,
    ReleaseBundleIntegrityError,
    ReleaseBundlePolicyError,
    _manifest_bytes_from_value,
)
from eval_factory.dataset.export import (
    ReleasePublicationItemCompilation,
    ReleasePublicationManifestCompilation,
)


class _CountingFacade:
    def __init__(self, delegate: RegistryLHWorkspaceExportFacade) -> None:
        self._delegate = delegate
        self.calls = 0

    def export(
        self,
        request: LHWorkspaceExportRequestV2,
        *,
        destination: Path,
    ) -> LHWorkspaceExportResultV2:
        self.calls += 1
        return self._delegate.export(
            request,
            destination=destination,
        )


class _BlockedFacade:
    def export(
        self,
        request: LHWorkspaceExportRequestV2,
        *,
        destination: Path,
    ) -> LHWorkspaceExportResultV2:
        del destination
        return LHWorkspaceExportResultV2.blocked(
            request=request,
            failure_code=LHWorkspaceExportFailureCodeV2.MATERIAL_NOT_FOUND,
        )


def _store(
    tmp_path: Path,
    *,
    fault_injector=None,
) -> NonProductionReleaseBundleStore:
    return NonProductionReleaseBundleStore(
        canary_root=tmp_path / "canary",
        internal_review_root=tmp_path / "internal",
        fault_injector=fault_injector,
    )


def _empty_facade() -> RegistryLHWorkspaceExportFacade:
    return RegistryLHWorkspaceExportFacade(resolver=MappingLHWorkspaceExportMaterialResolver(outputs={}))


def _file_compilation(
    compilation: ReleasePublicationManifestCompilation,
    source: Path,
) -> tuple[
    ReleasePublicationManifestCompilation,
    RegistryLHWorkspaceExportFacade,
]:
    item = compilation.items[0]
    digest = sha256_path(source)
    output_ref = FacadeObjectRef(
        object_type="attachment-output",
        object_id="attachment-output://r7-09/bundle-file",
        object_version="v2",
        object_sha256=digest,
    )
    member = LHWorkspaceExportMemberV2.create(
        normalized_path="inputs/source.txt",
        member_type=LHWorkspaceExportMemberTypeV2.FILE,
        media_type="text/plain",
        size_bytes=source.stat().st_size,
        content_sha256=digest,
        container_ref=None,
        output_ref=output_ref,
    )
    prior = item.item_manifest
    item_manifest = LHReleaseItemManifestV2.create(
        job_id=prior.job_id,
        item_id=prior.item_id,
        approved_release_result_ref=prior.approved_release_result_ref,
        approved_evaluation_item_ref=prior.approved_evaluation_item_ref,
        approved_release_decision_ref=prior.approved_release_decision_ref,
        release_subject_ref=prior.release_subject_ref,
        query_spec_ref=prior.query_spec_ref,
        rubric_set_ref=prior.rubric_set_ref,
        environment_spec_ref=prior.environment_spec_ref,
        provenance_manifest_ref=prior.provenance_manifest_ref,
        quality_report_ref=prior.quality_report_ref,
        final_package_manifest_ref=prior.final_package_manifest_ref,
        package_sha256=prior.package_sha256,
        query_yaml_sha256=prior.query_yaml_sha256,
        rubrics_json_sha256=prior.rubrics_json_sha256,
        expected_workspace_members=(member,),
        channel=prior.channel,
        registry=prior.registry,
        audit=prior.audit,
    )
    request = LHWorkspaceExportRequestV2.create(
        item_id=prior.item_id,
        final_package_manifest_ref=FacadeObjectRef(
            object_type=prior.final_package_manifest_ref.object_type,
            object_id=prior.final_package_manifest_ref.object_id,
            object_version=prior.final_package_manifest_ref.object_version,
            object_sha256=prior.final_package_manifest_ref.object_sha256,
        ),
        package_sha256=prior.package_sha256,
        output_refs=(output_ref,),
        members=(member,),
        max_member_count=100,
        max_total_bytes=1_000_000,
    )
    item_compilation = ReleasePublicationItemCompilation(
        source=item.source,
        item_manifest=item_manifest,
        query_yaml_bytes=item.query_yaml_bytes,
        rubrics_json_bytes=item.rubrics_json_bytes,
        workspace_request=request,
    )
    release = compilation.release_manifest
    manifest = NonProductionReleaseManifestV2.create(
        job_id=release.job_id,
        approved_item_ids=release.approved_item_ids,
        rejected_item_ids=release.rejected_item_ids,
        item_manifests=(item_manifest,),
        channel=release.channel,
        registry=release.registry,
        base_contract_manifest_ref=release.base_contract_manifest_ref,
        overlay_contract_manifest_ref=release.overlay_contract_manifest_ref,
        release_profile_decision_ref=release.release_profile_decision_ref,
        policy_ref=release.policy_ref,
        audit=release.audit,
    )
    return (
        ReleasePublicationManifestCompilation(
            release_manifest=manifest,
            items=(item_compilation,),
            rejected_item_ids=compilation.rejected_item_ids,
        ),
        RegistryLHWorkspaceExportFacade(
            resolver=MappingLHWorkspaceExportMaterialResolver(outputs={output_ref.object_id: source})
        ),
    )


def _empty_directory_compilation(
    compilation: ReleasePublicationManifestCompilation,
    source: Path,
) -> tuple[
    ReleasePublicationManifestCompilation,
    RegistryLHWorkspaceExportFacade,
]:
    item = compilation.items[0]
    output_ref = FacadeObjectRef(
        object_type="attachment-output",
        object_id="attachment-output://r7-09/empty-directory",
        object_version="v2",
        object_sha256=sha256_path(source),
    )
    member = LHWorkspaceExportMemberV2.create(
        normalized_path="inputs/empty",
        member_type=LHWorkspaceExportMemberTypeV2.DIRECTORY,
        media_type=None,
        size_bytes=0,
        content_sha256=None,
        container_ref=None,
        output_ref=output_ref,
    )
    prior = item.item_manifest
    item_manifest = LHReleaseItemManifestV2.create(
        job_id=prior.job_id,
        item_id=prior.item_id,
        approved_release_result_ref=prior.approved_release_result_ref,
        approved_evaluation_item_ref=prior.approved_evaluation_item_ref,
        approved_release_decision_ref=prior.approved_release_decision_ref,
        release_subject_ref=prior.release_subject_ref,
        query_spec_ref=prior.query_spec_ref,
        rubric_set_ref=prior.rubric_set_ref,
        environment_spec_ref=prior.environment_spec_ref,
        provenance_manifest_ref=prior.provenance_manifest_ref,
        quality_report_ref=prior.quality_report_ref,
        final_package_manifest_ref=prior.final_package_manifest_ref,
        package_sha256=prior.package_sha256,
        query_yaml_sha256=prior.query_yaml_sha256,
        rubrics_json_sha256=prior.rubrics_json_sha256,
        expected_workspace_members=(member,),
        channel=prior.channel,
        registry=prior.registry,
        audit=prior.audit,
    )
    request = LHWorkspaceExportRequestV2.create(
        item_id=prior.item_id,
        final_package_manifest_ref=FacadeObjectRef(
            object_type=prior.final_package_manifest_ref.object_type,
            object_id=prior.final_package_manifest_ref.object_id,
            object_version=prior.final_package_manifest_ref.object_version,
            object_sha256=prior.final_package_manifest_ref.object_sha256,
        ),
        package_sha256=prior.package_sha256,
        output_refs=(output_ref,),
        members=(member,),
        max_member_count=100,
        max_total_bytes=1_000_000,
    )
    item_compilation = replace(
        item,
        item_manifest=item_manifest,
        workspace_request=request,
    )
    release = compilation.release_manifest
    manifest = NonProductionReleaseManifestV2.create(
        job_id=release.job_id,
        approved_item_ids=release.approved_item_ids,
        rejected_item_ids=release.rejected_item_ids,
        item_manifests=(item_manifest,),
        channel=release.channel,
        registry=release.registry,
        base_contract_manifest_ref=release.base_contract_manifest_ref,
        overlay_contract_manifest_ref=release.overlay_contract_manifest_ref,
        release_profile_decision_ref=release.release_profile_decision_ref,
        policy_ref=release.policy_ref,
        audit=release.audit,
    )
    return (
        replace(
            compilation,
            release_manifest=manifest,
            items=(item_compilation,),
        ),
        RegistryLHWorkspaceExportFacade(
            resolver=MappingLHWorkspaceExportMaterialResolver(outputs={output_ref.object_id: source})
        ),
    )


def test_bundle_roots_must_be_nonproduction_and_nonoverlapping(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReleaseBundlePolicyError, match="overlapping"):
        NonProductionReleaseBundleStore(
            canary_root=tmp_path / "shared",
            internal_review_root=tmp_path / "shared",
        )
    with pytest.raises(ReleaseBundlePolicyError, match="overlapping"):
        NonProductionReleaseBundleStore(
            canary_root=tmp_path / "parent",
            internal_review_root=tmp_path / "parent/child",
        )
    with pytest.raises(ReleaseBundlePolicyError, match="production"):
        NonProductionReleaseBundleStore(
            canary_root=tmp_path / "production",
            internal_review_root=tmp_path / "internal",
        )
    physical = tmp_path / "physical"
    physical.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(physical, target_is_directory=True)
    with pytest.raises(ReleaseBundlePolicyError, match="real directory"):
        NonProductionReleaseBundleStore(
            canary_root=linked / "canary",
            internal_review_root=tmp_path / "internal",
        )


def test_builds_exact_empty_workspace_bundle(tmp_path: Path) -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)

    result = _store(tmp_path).build(
        compilation,
        facade=_empty_facade(),
    )[0]

    assert result.reused is False
    assert result.facts.bundle_file_count == 3
    assert result.bundle_path.is_dir()
    assert {
        path.relative_to(result.bundle_path).as_posix()
        for path in result.bundle_path.rglob("*")
        if path.is_file()
    } == {
        ".eval/rubrics.json",
        "query.yaml",
        "release-manifest.json",
    }
    assert (result.bundle_path / "workspace").is_dir()


def test_physical_release_manifest_excludes_audit_variation() -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)
    manifest = compilation.release_manifest
    changed_item = manifest.item_manifests[0].model_copy(
        update={
            "audit": manifest.item_manifests[0].audit.model_copy(
                update={"created_by": "different-item-auditor"}
            )
        }
    )
    changed = manifest.model_copy(
        update={
            "item_manifests": (changed_item,),
            "audit": manifest.audit.model_copy(update={"created_by": "different-manifest-auditor"}),
        }
    )

    assert _manifest_bytes_from_value(changed) == (_manifest_bytes_from_value(manifest))


def test_builds_file_bundle_and_exact_replay_skips_facade(
    tmp_path: Path,
) -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)
    attachment = tmp_path / "source.txt"
    attachment.write_text("safe input\n", encoding="utf-8")
    compilation, physical = _file_compilation(
        compilation,
        attachment,
    )
    facade = _CountingFacade(physical)
    store = _store(tmp_path)

    first = store.build(compilation, facade=facade)[0]
    second = store.build(compilation, facade=facade)[0]

    assert facade.calls == 1
    assert first.reused is False
    assert second.reused is True
    assert first.bundle_path == second.bundle_path
    assert (first.bundle_path / "workspace/inputs/source.txt").read_text() == "safe input\n"
    assert (
        store.verify_item(
            compilation.release_manifest,
            compilation.items[0],
            first.facts,
        )
        == first.bundle_path
    )


def test_bundle_preserves_manifested_empty_workspace_directory(
    tmp_path: Path,
) -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)
    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()
    compilation, facade = _empty_directory_compilation(
        compilation,
        empty_directory,
    )

    result = _store(tmp_path).build(
        compilation,
        facade=facade,
    )[0]

    assert (result.bundle_path / "workspace/inputs/empty").is_dir()
    assert result.facts.bundle_file_count == 3


def test_after_rename_crash_leaves_reusable_orphan(
    tmp_path: Path,
) -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)
    calls: list[str] = []

    def fault(point: str) -> None:
        calls.append(point)
        if point == "after_rename":
            raise RuntimeError("simulated crash")

    facade = _CountingFacade(_empty_facade())
    with pytest.raises(RuntimeError, match="simulated crash"):
        _store(tmp_path, fault_injector=fault).build(
            compilation,
            facade=facade,
        )

    replay = _store(tmp_path).build(
        compilation,
        facade=facade,
    )[0]
    assert "after_rename" in calls
    assert facade.calls == 1
    assert replay.reused is True


def test_bundle_corruption_and_unexpected_symlink_fail_closed(
    tmp_path: Path,
) -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)
    store = _store(tmp_path)
    result = store.build(compilation, facade=_empty_facade())[0]
    (result.bundle_path / "query.yaml").write_text(
        "changed",
        encoding="utf-8",
    )
    with pytest.raises(ReleaseBundleIntegrityError, match="differ"):
        store.build(compilation, facade=_empty_facade())

    second_root = tmp_path / "second"
    second = _store(second_root)
    clean = second.build(compilation, facade=_empty_facade())[0]
    (clean.bundle_path / "workspace/link").symlink_to(clean.bundle_path / "query.yaml")
    with pytest.raises(ReleaseBundleIntegrityError, match="symlink"):
        second.build(compilation, facade=_empty_facade())


def test_blocked_workspace_export_creates_no_bundle(
    tmp_path: Path,
) -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)
    store = _store(tmp_path)

    with pytest.raises(ReleaseBundleExportError, match="MATERIAL_NOT_FOUND"):
        store.build(
            compilation,
            facade=_BlockedFacade(),
        )

    assert not any(
        path.is_dir() and len(path.name) == 64 for path in (tmp_path / "canary/cas/sha256").rglob("*")
    )


def test_verify_rejects_changed_bundle_facts(tmp_path: Path) -> None:
    source, _ = _source()
    compilation = _compile_manifest(source)
    store = _store(tmp_path)
    result = store.build(compilation, facade=_empty_facade())[0]
    stale = replace(
        result.facts,
        bundle_total_bytes=result.facts.bundle_total_bytes + 1,
    )

    with pytest.raises(ReleaseBundleIntegrityError, match="facts differ"):
        store.verify_item(
            compilation.release_manifest,
            compilation.items[0],
            stale,
        )


def test_bundle_store_exposes_no_enumeration_or_production_api() -> None:
    public = set(NonProductionReleaseBundleStore.__dict__)
    assert "list" not in public
    assert "production_root" not in public
    assert "resolve_production" not in public
