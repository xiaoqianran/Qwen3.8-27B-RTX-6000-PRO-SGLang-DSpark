"""Prepare the pinned Pi LoRA adapter in the same Modal model Volume.

One explicit CPU job populates /models/adapters/pi-lora. Serving mounts only
/models read-only, and never downloads adapters during a GPU cold start.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

from deploy.modal_config import (
    CONFIG, MODEL_STORE_PATH, PI_LORA_PATH, PI_LORA_REPO_ID, PI_LORA_REVISION,
)

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
ADAPTER_MANIFEST = ".adapter-manifest.json"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_pi_lora(folder: str = PI_LORA_PATH, *, verify_checksum: bool = False) -> dict:
    """Reject incomplete, stale or misplaced adapters before GPU allocation."""
    root = Path(folder)
    marker = root / ADAPTER_MANIFEST
    if not marker.is_file():
        raise RuntimeError(
            f"Pi LoRA manifest missing: {marker}. "
            "Run python -m deploy.pi_lora --execute"
        )
    record = json.loads(marker.read_text(encoding="utf-8"))
    if (record.get("version") != 1 or record.get("repo_id") != PI_LORA_REPO_ID
            or record.get("revision") != PI_LORA_REVISION):
        raise RuntimeError("Pi LoRA repository or pinned revision mismatch")
    files = record.get("files")
    if not isinstance(files, dict) or set(files) != set(ADAPTER_FILES):
        raise RuntimeError("Pi LoRA manifest files are incomplete")
    for name in ADAPTER_FILES:
        expected = files[name]
        if (not isinstance(expected, dict)
                or not isinstance(expected.get("size"), int)
                or expected["size"] <= 0
                or not isinstance(expected.get("sha256"), str)
                or len(expected["sha256"]) != 64):
            raise RuntimeError(f"Invalid Pi LoRA manifest entry: {name}")
        path = root / name
        if not path.is_file() or path.stat().st_size != expected["size"]:
            raise RuntimeError(f"Missing or incomplete Pi LoRA file: {path}")
        if verify_checksum and _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Pi LoRA SHA256 mismatch: {path}")
    return record


def _prepare_adapter(*, download_file, file_info: dict, volume, mount: str) -> dict:
    """Stage a complete adapter, then atomically expose it via one commit."""
    destination = Path(mount) / "adapters" / "pi-lora"
    if destination.exists():
        return {"status": "REUSED", "manifest": validate_pi_lora(
            str(destination), verify_checksum=True)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / (".pi-lora-" + uuid.uuid4().hex)
    temporary.mkdir()
    try:
        metadata = {}
        for name in ADAPTER_FILES:
            source = Path(download_file(name))
            target = temporary / name
            shutil.copyfile(source, target)
            digest = _sha256(target)
            expected_sha = file_info[name].get("sha256")
            expected_size = file_info[name]["size"]
            if target.stat().st_size != expected_size or (
                expected_sha is not None and digest != expected_sha
            ):
                raise RuntimeError(f"Pi LoRA source integrity mismatch: {name}")
            metadata[name] = {"size": expected_size, "sha256": digest}
        record = {"version": 1, "repo_id": PI_LORA_REPO_ID,
                  "revision": PI_LORA_REVISION, "files": metadata}
        (temporary / ADAPTER_MANIFEST).write_text(
            json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
        )
        validate_pi_lora(str(temporary), verify_checksum=True)
        # Existing adapters cannot be replaced silently while a serving process
        # might be reading them. An adapter revision change needs a deliberate
        # migration, not an in-place overwrite.
        if destination.exists():
            raise RuntimeError(f"Pi LoRA destination already exists: {destination}")
        os.rename(temporary, destination)
        try:
            volume.commit()
        except Exception:
            # A failed commit must not leave a locally visible completed
            # snapshot that a retry could mistake for a persisted adapter.
            shutil.rmtree(destination)
            raise
        return {"status": "PUBLISHED", "manifest": record}
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _download_on_modal(volume) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().repo_info(
        PI_LORA_REPO_ID, revision=PI_LORA_REVISION, files_metadata=True
    )
    if info.sha != PI_LORA_REVISION:
        raise RuntimeError("Hugging Face Pi LoRA revision unexpectedly changed")
    entries = {file.rfilename: file for file in info.siblings}
    metadata = {}
    for name in ADAPTER_FILES:
        if name not in entries or not entries[name].size:
            raise RuntimeError(f"Missing Pi LoRA source file: {name}")
        lfs = entries[name].lfs
        metadata[name] = {
            "size": entries[name].size,
            "sha256": lfs.sha256 if lfs is not None else None,
        }
    return _prepare_adapter(
        download_file=lambda name: hf_hub_download(
            repo_id=PI_LORA_REPO_ID, filename=name, revision=PI_LORA_REVISION
        ),
        file_info=metadata, volume=volume, mount=MODEL_STORE_PATH,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Start one Modal CPU job")
    args = parser.parse_args(argv)
    print("PLAN " + json.dumps({
        "volume": CONFIG.model_volume_name, "path": PI_LORA_PATH,
        "repo_id": PI_LORA_REPO_ID, "revision": PI_LORA_REVISION,
        "files": ADAPTER_FILES, "execute": args.execute,
    }), flush=True)
    if not args.execute:
        return

    import modal

    volume = modal.Volume.from_name(CONFIG.model_volume_name)
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("huggingface_hub==1.33.0", "hf_xet==1.8.0")
             .env({"HF_HOME": "/tmp/hf-cache", "HF_HUB_DISABLE_XET": "0"})
             .add_local_python_source("deploy", copy=True, ignore=[]))
    app = modal.App("qwen38-pi-lora-prepare")

    @app.function(
        image=image, volumes={MODEL_STORE_PATH: volume}, serialized=True,
        cpu=4, memory=8192, timeout=1800,
    )
    def prepare() -> dict:
        return _download_on_modal(volume)

    with app.run():
        result = prepare.remote()
    # SDK readback is independent of the writer's mounted filesystem.
    expected = result["manifest"]
    marker = "/adapters/pi-lora/" + ADAPTER_MANIFEST
    committed = json.loads(b"".join(volume.read_file(marker)).decode("utf-8"))
    if committed != expected:
        raise RuntimeError("Pi LoRA Volume publication readback mismatch")
    print("RESULT " + json.dumps({
        "status": result["status"], "volume": CONFIG.model_volume_name,
        "path": PI_LORA_PATH, "revision": PI_LORA_REVISION,
        "bytes": sum(f["size"] for f in committed["files"].values()),
        "verified": True,
    }), flush=True)


if __name__ == "__main__":
    main()
