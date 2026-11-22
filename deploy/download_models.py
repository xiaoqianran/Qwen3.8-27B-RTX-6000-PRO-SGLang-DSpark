"""Prepare Qwen3.8 + optional DFlash2 in one Coordinator call (CPU only)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy.modal_config import CONFIG

MOUNT = "/models"
MANIFEST = ".modal-model-manifest.json"


def sources() -> list[dict[str, str]]:
    models = [{"repo_id": CONFIG.model_id,
               "revision": CONFIG.model_revision or "main"}]
    if CONFIG.speculative_algorithm != "NONE":
        models.append({"repo_id": CONFIG.draft_model_id,
                       "revision": CONFIG.draft_model_revision or "main"})
    return models


def _model_path(root: Path, entry: dict, expected_repo: str, expected_revision: str | None) -> Path:
    if entry.get("repo_id") != expected_repo:
        raise RuntimeError(f"Wrong model repository: expected {expected_repo}")
    revision = entry.get("revision")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(f"Missing resolved revision: {expected_repo}")
    # Immutable revision pins must match; branch references resolve at preparation.
    if (expected_revision and len(expected_revision) == 40
            and all(c in "0123456789abcdef" for c in expected_revision.lower())
            and expected_revision.lower() != revision.lower()):
        raise RuntimeError(f"Unexpected model revision: {expected_repo}")
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError("Published model path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise RuntimeError("Published model path escapes the model Volume")
    if not (path / "config.json").is_file():
        raise RuntimeError(f"Missing model config: {path}")
    indices = list(path.glob("*.safetensors.index.json"))
    if indices:
        shards = set()
        for index in indices:
            data = json.loads(index.read_text(encoding="utf-8"))
            shards.update(data.get("weight_map", {}).values())
        if not shards or any(not (path / shard).is_file() for shard in shards):
            raise RuntimeError(f"Missing indexed model shards: {path}")
    elif not any(path.glob("*.safetensors")):
        raise RuntimeError(f"No model weight files: {path}")
    return path


def validate_model_store(
    root_dir: str,
    target_repo: str,
    draft_repo: str,
    *,
    target_revision: str | None = None,
    draft_revision: str | None = None,
    require_draft: bool = True,
) -> dict[str, str]:
    """Read the published model revision map; never download on GPU."""
    root = Path(root_dir).resolve()
    marker = root / MANIFEST
    if not marker.is_file():
        raise RuntimeError("Model cache missing: run python -m deploy.download_models --execute")
    record = json.loads(marker.read_text(encoding="utf-8"))
    if record.get("version") != 2:
        raise RuntimeError("Expected v2 download manifest; prepare models with modal-benchmark")
    result = {"target": str(_model_path(root, record["target"], target_repo, target_revision))}
    if require_draft:
        result["draft"] = str(_model_path(root, record["draft"], draft_repo, draft_revision))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Execute real Modal CPU download")
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=2)
    args = parser.parse_args(argv)

    models = sources()
    aliases = {"target": CONFIG.model_id}
    if CONFIG.speculative_algorithm != "NONE":
        aliases["draft"] = CONFIG.draft_model_id
    print("PLAN " + json.dumps({
        "models": models, "roles": aliases,
        "volume": CONFIG.model_volume_name,
        "workers": args.workers, "verification": "xet",
        "execute": args.execute,
    }, ensure_ascii=False), flush=True)
    if not args.execute:
        return

    # No extra publish function: Coordinator verifies files and atomically
    # writes the same v2 manifest consumed by validate_model_store().
    from modal_benchmark.hf import create_downloader
    downloader = create_downloader(
        volume_name=CONFIG.model_volume_name, volume_version=None,
        mount_path=MOUNT, max_workers=args.workers,
        verification_mode="xet",
    )
    result = downloader.download(
        models, workers=args.workers, active_manifest=MANIFEST, aliases=aliases)
    if result.get("status") != "VERIFIED":
        raise RuntimeError("Qwen3.8 model preparation was not verified")
    print("PUBLISHED " + json.dumps({
        "models": result["models"], "timings": result["timings"],
        "seconds": result["end_to_end_seconds"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
