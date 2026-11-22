"""Filesystem model store: one immutable directory per version plus an active pointer.

Layout::

    <root>/
      registry.json          {"active": "v0002", "versions": ["v0001", "v0002"]}
      v0001/model.joblib
      v0001/metadata.json

MLflow records the experiment history. This store is the serving contract: the API
only needs a directory, so it does not depend on a tracking server at runtime.
Version switching and rollback build on ``set_active`` (next milestone).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

DEFAULT_ROOT = Path("artifacts/models")
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REGISTRY_FILE = "registry.json"
_VERSION_RE = re.compile(r"^v\d{4}$")


@dataclass(frozen=True)
class LoadedModel:
    version: str
    pipeline: Any
    metadata: dict[str, Any]


def store_root() -> Path:
    return Path(os.environ.get("CHURN_MODEL_DIR", DEFAULT_ROOT))


def _read_registry(root: Path) -> dict[str, Any]:
    path = root / REGISTRY_FILE
    if not path.exists():
        return {"active": None, "versions": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_registry(root: Path, registry: dict[str, Any]) -> None:
    tmp = root / (REGISTRY_FILE + ".tmp")
    tmp.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    tmp.replace(root / REGISTRY_FILE)  # atomic swap so readers never see a partial file


def list_versions(root: Path) -> list[str]:
    return list(_read_registry(root)["versions"])


def active_version(root: Path) -> str | None:
    return _read_registry(root)["active"]


def save_model(pipeline: Any, metadata: dict[str, Any], root: Path, activate: bool = True) -> str:
    """Persist a new immutable version and optionally mark it active."""
    root.mkdir(parents=True, exist_ok=True)
    registry = _read_registry(root)
    version = f"v{len(registry['versions']) + 1:04d}"
    version_dir = root / version
    version_dir.mkdir(exist_ok=False)
    joblib.dump(pipeline, version_dir / MODEL_FILE)
    full_metadata = {"version": version, **metadata}
    (version_dir / METADATA_FILE).write_text(
        json.dumps(full_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    registry["versions"].append(version)
    if activate:
        registry["active"] = version
    _write_registry(root, registry)
    return version


def set_active(version: str, root: Path) -> None:
    registry = _read_registry(root)
    if version not in registry["versions"]:
        raise KeyError(f"unknown model version {version!r}; known: {registry['versions']}")
    registry["active"] = version
    _write_registry(root, registry)


def load_model(root: Path, version: str | None = None) -> LoadedModel:
    """Load ``version`` or, by default, the active version."""
    version = version or active_version(root)
    if version is None:
        raise FileNotFoundError(f"no active model in {root}; run `churn-train` first")
    if not _VERSION_RE.match(version):
        raise ValueError(f"invalid version string {version!r}")
    version_dir = root / version
    metadata = json.loads((version_dir / METADATA_FILE).read_text(encoding="utf-8"))
    pipeline = joblib.load(version_dir / MODEL_FILE)
    return LoadedModel(version=version, pipeline=pipeline, metadata=metadata)
