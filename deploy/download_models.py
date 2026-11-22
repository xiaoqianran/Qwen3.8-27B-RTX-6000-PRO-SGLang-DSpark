"""Download Qwen3.8 target and DFlash2 draft using modal-benchmark.

Usage: uv run python -m deploy.download_models --execute
No GPU is allocated by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

from deploy.modal_config import CONFIG

MOUNT = "/models"
MANIFEST = ".modal-model-manifest.json"


def sources() -> list[dict[str, str]]:
    result = [{"repo_id": CONFIG.model_id, "revision": CONFIG.model_revision or "main"}]
    if CONFIG.speculative_algorithm != "NONE":
        result.append({"repo_id": CONFIG.draft_model_id,
                       "revision": CONFIG.draft_model_revision or "main"})
    return result


def published_record(manifest: dict, base: str, root: str = MOUNT) -> dict:
    """Translate benchmark paths to relative paths in the mounted Volume."""
    models = {entry["repo_id"]: entry for entry in manifest["models"]}
    if len(models) != len(manifest["models"]):
        raise ValueError("Duplicate repository in download manifest")
    result = {"version": 2, "downloader": "modal-benchmark-0.1.5"}
    # Publish what the verified immutable manifest actually contains; the
    # publisher's remote environment need not match the local CLI environment.
    roles = [("target", CONFIG.model_id)]
    if CONFIG.draft_model_id in models:
        roles.append(("draft", CONFIG.draft_model_id))
    for role, repo_id in roles:
        entry = models[repo_id]
        revision = entry["revision"]
        location = PurePosixPath(base) / repo_id / revision / "model"
        if not location.is_relative_to(PurePosixPath(root)):
            raise ValueError("Model path outside mounted Volume")
        result[role] = {
            "repo_id": repo_id,
            "revision": revision,
            "path": location.relative_to(PurePosixPath(root)).as_posix(),
        }
    return result


def check_completion_receipt(manifest: dict, record: dict, root: Path) -> None:
    """Validate benchmark completion at revision root, outside its model/ dir."""
    folder = root / record["target"]["path"]
    marker = folder.parent / "all-components-complete.json"
    if not marker.is_file():
        raise RuntimeError(f"Benchmark completion marker missing: {marker}")
    receipt = json.loads(marker.read_text(encoding="utf-8"))
    expected = {"status": "COMPLETE", "manifest": manifest, "verification_mode": "xet"}
    if receipt != expected:
        raise RuntimeError("Benchmark completion receipt does not match verified manifest")


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
    parser.add_argument("--execute", action="store_true", help="Run real CPU download")
    parser.add_argument("--workers", type=int, choices=range(1, 6), default=2)
    args = parser.parse_args(argv)
    plan = {
        "sources": sources(), "volume_name": CONFIG.model_volume_name,
        "workers": args.workers, "verification_mode": "xet", "volume_version": "existing",
    }
    print("PLAN " + json.dumps(plan, ensure_ascii=False), flush=True)
    if not args.execute:
        return

    import modal
    from modal_benchmark.hf import create_downloader
    downloader = create_downloader(
        volume_name=CONFIG.model_volume_name,
        volume_version=None,  # retain the existing Volume version
        mount_path=MOUNT,
        max_workers=args.workers,
        verification_mode="xet",
        region=None,
    )
    app = downloader.app
    volume = downloader.volume

    @app.function(
        image=modal.Image.debian_slim(python_version="3.12").add_local_python_source("deploy"),
        serialized=True,
        volumes={MOUNT: volume},
        cpu=1,
        timeout=300,
    )
    def publish(manifest: dict, base: str) -> dict:
        volume.reload()
        record = published_record(manifest, base)
        root = Path(MOUNT)
        for role in ("target", "draft"):
            if role not in record:
                continue
            model = root / record[role]["path"]
            if not (model / "config.json").is_file():
                raise RuntimeError(f"{role} config.json is missing: {model}")
            if not any(model.rglob("*.safetensors")):
                raise RuntimeError(f"{role} weights are missing: {model}")
        # Reader has verified all models before publication; verify its completion
        # receipt still exists in the newly reloaded Volume.
        check_completion_receipt(manifest, record, root)
        dest = root / MANIFEST
        temporary = dest.with_name(f"{MANIFEST}.{os.getpid()}.tmp")
        try:
            temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
            os.replace(temporary, dest)
            volume.commit()
        finally:
            temporary.unlink(missing_ok=True)
        return record

    with app.run():
        manifest = downloader.discover_many(sources())
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
        base = f"{MOUNT}/downloads/full/{identity}"
        outcome = downloader.run_manifest(manifest, base, workers=str(args.workers))
        if outcome.get("status") != "VERIFIED":
            raise RuntimeError("Download did not complete verification")
        record = publish.remote(manifest, base)
        print("PUBLISHED " + json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
