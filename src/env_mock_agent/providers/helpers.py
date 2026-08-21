from __future__ import annotations

import hashlib
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

from env_mock_agent.schemas import ArtifactResult, ArtifactStatus

CORE_PROPERTIES_PATH = "docProps/core.xml"
CORE_PROPERTY_NAMESPACES = {
    "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "dcmitype": "http://purl.org/dc/dcmitype/",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def artifact_result(
    artifact_id: str,
    path: Path,
    provider: str,
    *,
    metadata: dict[str, object] | None = None,
) -> ArtifactResult:
    bytes_count = (
        path.stat().st_size
        if path.is_file()
        else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    )
    return ArtifactResult(
        artifact_id=artifact_id,
        status=ArtifactStatus.BUILT,
        path=str(path),
        sha256=sha256_path(path),
        bytes_count=bytes_count,
        runtime=provider,
        attempts=1,
        metadata=metadata or {},
    )


def as_string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def _normalize_core_properties(content: bytes, timestamp: datetime) -> bytes:
    for prefix, uri in CORE_PROPERTY_NAMESPACES.items():
        ElementTree.register_namespace(prefix, uri)
    root = ElementTree.fromstring(content)
    value = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    namespace = CORE_PROPERTY_NAMESPACES["dcterms"]
    for name in ("created", "modified"):
        element = root.find(f"{{{namespace}}}{name}")
        if element is not None:
            element.text = value
    return cast(bytes, ElementTree.tostring(root, encoding="utf-8", xml_declaration=False))


def normalize_zip_timestamps(path: Path, seed: int) -> None:
    normalized = path.with_suffix(f"{path.suffix}.normalized")
    timestamp = datetime(2020, 1, 1) + timedelta(days=seed % 366)
    zip_timestamp = (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        0,
        0,
        0,
    )
    with (
        zipfile.ZipFile(path, "r") as source,
        zipfile.ZipFile(
            normalized,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as destination,
    ):
        for name in sorted(source.namelist()):
            original = source.getinfo(name)
            info = zipfile.ZipInfo(name, date_time=zip_timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = original.external_attr
            info.create_system = original.create_system
            content = source.read(name)
            if name == CORE_PROPERTIES_PATH:
                content = _normalize_core_properties(content, timestamp)
            destination.writestr(info, content)
    normalized.replace(path)
