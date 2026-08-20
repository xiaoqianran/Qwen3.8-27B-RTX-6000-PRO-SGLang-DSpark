"""CPU-only model preparation helpers for the Modal backend."""

from __future__ import annotations


def download_models(
    *,
    target_repo: str,
    target_revision: str | None,
    target_dir: str,
    draft_repo: str,
    draft_revision: str | None,
    draft_dir: str,
    model_store_dir: str,
    max_workers: int,
) -> None:
    """Download both model repositories into the persistent model Volume."""
    import json
    from pathlib import Path

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest: dict[str, dict[str, str | None]] = {}

    for role, repo_id, revision, local_dir in (
        ("target", target_repo, target_revision, target_dir),
        ("draft", draft_repo, draft_revision, draft_dir),
    ):
        resolved_revision = api.model_info(repo_id, revision=revision).sha
        print(f"[CPU] downloading {role}: {repo_id}@{resolved_revision}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=resolved_revision,
            local_dir=local_dir,
            max_workers=max_workers,
        )

        if not (Path(local_dir) / "config.json").is_file():
            raise RuntimeError(f"incomplete {role} model at {local_dir}")

        manifest[role] = {
            "repo_id": repo_id,
            "revision": resolved_revision,
        }

    Path(model_store_dir, ".modal-model-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[CPU] model Volume ready", flush=True)


def validate_model_store(
    model_store_dir: str,
    target_dir: str,
    draft_dir: str,
    target_repo: str,
    draft_repo: str,
) -> None:
    """Fail closed if metadata or model weight shards are incomplete."""
    import json
    from pathlib import Path

    manifest_path = Path(model_store_dir, ".modal-model-manifest.json")
    required = (
        Path(target_dir, "config.json"),
        Path(draft_dir, "config.json"),
        manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("model Volume is incomplete: " + ", ".join(missing))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"target": target_repo, "draft": draft_repo}
    for role, repo_id in expected.items():
        if (manifest.get(role) or {}).get("repo_id") != repo_id:
            raise RuntimeError(f"model Volume contains the wrong {role} repository")

    for role, directory in (("target", Path(target_dir)), ("draft", Path(draft_dir))):
        index_files = list(directory.glob("*.safetensors.index.json"))
        if index_files:
            shard_names: set[str] = set()
            for index_path in index_files:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shard_names.update((index.get("weight_map") or {}).values())
            missing_shards = [name for name in shard_names if not (directory / name).is_file()]
            if missing_shards:
                raise RuntimeError(
                    f"{role} model is missing {len(missing_shards)} safetensors shard(s)"
                )
        elif not any(directory.glob("*.safetensors")):
            raise RuntimeError(f"{role} model has no safetensors weights at {directory}")
