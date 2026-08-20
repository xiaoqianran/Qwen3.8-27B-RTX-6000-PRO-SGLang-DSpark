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
        print(f"[CPU] downloading {role}: {repo_id}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            max_workers=max_workers,
        )

        if not (Path(local_dir) / "config.json").is_file():
            raise RuntimeError(f"incomplete {role} model at {local_dir}")

        manifest[role] = {
            "repo_id": repo_id,
            "revision": api.model_info(repo_id, revision=revision).sha,
        }

    Path(model_store_dir, ".modal-model-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[CPU] model Volume ready", flush=True)


def validate_model_store(
    model_store_dir: str,
    target_dir: str,
    draft_dir: str,
) -> None:
    """Fail closed if the model Volume is incomplete."""
    from pathlib import Path

    required = (
        Path(target_dir, "config.json"),
        Path(draft_dir, "config.json"),
        Path(model_store_dir, ".modal-model-manifest.json"),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("model Volume is incomplete: " + ", ".join(missing))
