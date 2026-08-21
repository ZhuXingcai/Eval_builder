from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from test_production_release import _payload, _policy
from test_production_release_persistence import (
    _production_service,
    _publish_production,
)

from eval_factory.dataset.production_release import (
    ProductionReleaseCompiler,
    ProductionReleaseConflictError,
    ProductionReleaseIntegrityError,
)
from eval_factory.dataset.production_release_store import (
    ProductionRegistryStore,
    ProductionReleaseReportStore,
)


def test_repository_store_first_authority_and_exact_replay(
    tmp_path: Path,
) -> None:
    payload = _payload()
    policy = _policy(payload)
    compilation = ProductionReleaseCompiler().compile_repository_blocked(
        payload=payload,
        policy=policy,
    )
    store = ProductionReleaseReportStore(
        tmp_path / "reports",
        max_report_bytes=policy.max_report_bytes,
    )
    request_sha = hashlib.sha256(policy.canonical_json() + payload).hexdigest()

    first = store.persist_repository(
        acceptance_key="production-release-acceptance://r8-09",
        request_sha256=request_sha,
        payload=payload,
        compilation=compilation,
    )
    before = _inventory(tmp_path / "reports")
    replay = store.persist_repository(
        acceptance_key="production-release-acceptance://r8-09",
        request_sha256=request_sha,
        payload=payload,
        compilation=compilation,
    )

    assert replay == first
    assert store.get_accepted_result(first.acceptance_key) == compilation.result
    assert _inventory(tmp_path / "reports") == before


def test_repository_store_rejects_changed_request_and_corruption(
    tmp_path: Path,
) -> None:
    payload = _payload()
    policy = _policy(payload)
    compilation = ProductionReleaseCompiler().compile_repository_blocked(
        payload=payload,
        policy=policy,
    )
    store = ProductionReleaseReportStore(
        tmp_path / "reports",
        max_report_bytes=policy.max_report_bytes,
    )
    accepted = store.persist_repository(
        acceptance_key="production-release-acceptance://r8-09",
        request_sha256="1" * 64,
        payload=payload,
        compilation=compilation,
    )
    with pytest.raises(ProductionReleaseConflictError, match="changed"):
        store.persist_repository(
            acceptance_key=accepted.acceptance_key,
            request_sha256="2" * 64,
            payload=payload,
            compilation=compilation,
        )

    envelope = next((tmp_path / "reports/envelopes/result").glob("sha256/*/*"))
    envelope.write_text("{}", encoding="utf-8")
    with pytest.raises(ProductionReleaseIntegrityError):
        store.get_accepted_result(accepted.acceptance_key)


def test_repository_store_rejects_invalid_root(tmp_path: Path) -> None:
    database = tmp_path / "not-a-directory"
    sqlite3.connect(database).close()
    with pytest.raises(ProductionReleaseIntegrityError, match="root"):
        ProductionReleaseReportStore(database, max_report_bytes=10_000)

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ProductionReleaseIntegrityError, match="root"):
        ProductionReleaseReportStore(
            linked / "reports",
            max_report_bytes=10_000,
        )
    with pytest.raises(ProductionReleaseIntegrityError, match="root"):
        ProductionRegistryStore(
            linked / "production/registry",
            max_registry_bytes=10_000,
        )


def test_repository_store_rejects_internal_symlink_members(
    tmp_path: Path,
) -> None:
    policy = _policy(_payload())
    envelope_store = ProductionReleaseReportStore(
        tmp_path / "envelope-store",
        max_report_bytes=policy.max_report_bytes,
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (envelope_store.root / "envelopes").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(ProductionReleaseIntegrityError, match="symlink"):
        envelope_store.put_policy(policy)

    cas_store = ProductionReleaseReportStore(
        tmp_path / "cas-store",
        max_report_bytes=policy.max_report_bytes,
    )
    (cas_store.root / "cas").rmdir()
    (cas_store.root / "cas").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(ProductionReleaseIntegrityError, match="symlink"):
        cas_store.put_policy(policy)


def test_store_limits_missing_authority_and_registry_scope(
    tmp_path: Path,
) -> None:
    payload = _payload()
    policy = _policy(payload)
    compilation = ProductionReleaseCompiler().compile_repository_blocked(
        payload=payload,
        policy=policy,
    )
    with pytest.raises(ProductionReleaseIntegrityError, match="limit"):
        ProductionReleaseReportStore(
            tmp_path / "invalid-limit",
            max_report_bytes=1,
        )

    store = ProductionReleaseReportStore(
        tmp_path / "reports",
        max_report_bytes=policy.max_report_bytes,
    )
    with pytest.raises(ProductionReleaseIntegrityError, match="acceptance"):
        store.get_accepted_result("production-release-acceptance://missing")
    with pytest.raises(ProductionReleaseIntegrityError, match="envelope"):
        store.get_policy(policy.to_ref())

    tiny = ProductionReleaseReportStore(
        tmp_path / "tiny",
        max_report_bytes=2,
    )
    with pytest.raises(ProductionReleaseIntegrityError, match="byte"):
        tiny.put_policy(policy)

    with pytest.raises(ProductionReleaseIntegrityError, match="production"):
        ProductionRegistryStore(
            tmp_path / "canary/registry",
            max_registry_bytes=policy.max_registry_bytes,
        )
    registry = ProductionRegistryStore(
        tmp_path / "production/registry",
        max_registry_bytes=policy.max_registry_bytes,
    )
    with pytest.raises(ProductionReleaseIntegrityError, match="published"):
        registry.put_closure(
            policy=policy,
            result=compilation.result,
        )


def test_production_registry_reuses_and_rejects_corrupt_closure(
    tmp_path: Path,
) -> None:
    (
        service,
        source,
        policy,
        _job_store,
        _bundle_store,
        registry_store,
        profile_payload,
    ) = _production_service(tmp_path)
    result = _publish_production(
        service,
        source,
        policy,
        profile_payload,
    )
    inventory = _inventory(registry_store.root)

    assert registry_store.get_result(result.to_ref()) == result
    assert (
        registry_store.put_closure(
            policy=policy,
            result=result,
        )
        == result.to_ref()
    )
    assert _inventory(registry_store.root) == inventory
    assert {path.name for path in (registry_store.root / "envelopes").iterdir() if path.is_dir()} == {
        "attestation-authority",
        "item-manifest",
        "policy",
        "published-projection",
        "receipt",
        "registry-entry",
        "release-manifest",
        "result",
    }

    envelope = next((registry_store.root / "envelopes/attestation-authority").glob("sha256/*/*"))
    envelope.write_text("{}", encoding="utf-8")
    with pytest.raises(ProductionReleaseIntegrityError):
        registry_store.get_result(result.to_ref())


def _inventory(root: Path) -> tuple[int, int]:
    files = tuple(path for path in root.rglob("*") if path.is_file())
    return len(files), sum(path.stat().st_size for path in files)
