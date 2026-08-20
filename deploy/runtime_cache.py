"""Versioned persistent cache helpers for GPU cold-start artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeCache:
    key: str
    root: str
    triton_dir: str
    torchinductor_dir: str
    sglang_dir: str
    flashinfer_entries_before: int

    @property
    def env(self) -> dict[str, str]:
        return {
            "TRITON_CACHE_DIR": self.triton_dir,
            "TORCHINDUCTOR_CACHE_DIR": self.torchinductor_dir,
            "SGLANG_CACHE_DIR": self.sglang_dir,
            "SGLANG_FLASHINFER_AUTOTUNE_CACHE": "1",
        }


def _read_model_manifest(model_store_dir: str) -> dict[str, Any]:
    path = Path(model_store_dir, ".modal-model-manifest.json")
    if not path.is_file():
        raise RuntimeError(f"model manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _flashinfer_entry_count(sglang_dir: str) -> int:
    root = Path(sglang_dir, "flashinfer", "autotune")
    return sum(1 for path in root.rglob("*.json") if path.is_file()) if root.exists() else 0


def prepare_runtime_cache(
    *,
    compile_cache_dir: str,
    model_store_dir: str,
    identity: dict[str, Any],
) -> RuntimeCache:
    """Create a cache namespace tied to the exact model/runtime configuration."""
    model_manifest = _read_model_manifest(model_store_dir)
    payload = {
        **identity,
        "target": model_manifest.get("target"),
        "draft": model_manifest.get("draft"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(encoded).hexdigest()[:16]

    root = Path(compile_cache_dir, "runtime", key)
    triton_dir = root / "triton"
    torchinductor_dir = root / "torchinductor"
    sglang_dir = root / "sglang"
    for directory in (triton_dir, torchinductor_dir, sglang_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(
            json.dumps({"cache_key": key, **payload}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return RuntimeCache(
        key=key,
        root=str(root),
        triton_dir=str(triton_dir),
        torchinductor_dir=str(torchinductor_dir),
        sglang_dir=str(sglang_dir),
        flashinfer_entries_before=_flashinfer_entry_count(str(sglang_dir)),
    )


def flashinfer_entry_count(cache: RuntimeCache) -> int:
    return _flashinfer_entry_count(cache.sglang_dir)
