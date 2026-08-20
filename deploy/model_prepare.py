"""CPU-only model preparation helpers for the Modal backend."""

from __future__ import annotations

import json
from pathlib import Path

from deploy.modal_config import DRAFT_MODEL_PATH, MODEL_STORE_PATH, TARGET_MODEL_PATH


def download_models(
    target_repo: str,
    target_revision: str | None,
    draft_repo: str,
    draft_revision: str | None,
    max_workers: int,
) -> None:
    """Download both model repositories into the persistent model Volume."""
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest = {}

    for role, repo_id, revision, local_dir in (
        ("target", target_repo, target_revision, TARGET_MODEL_PATH),
        ("draft", draft_repo, draft_revision, DRAFT_MODEL_PATH),
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

    Path(MODEL_STORE_PATH, ".modal-model-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[CPU] model Volume ready", flush=True)


def validate_model_store() -> None:
    """Fail closed if the GPU-visible model Volume is incomplete."""
    required = (
        Path(TARGET_MODEL_PATH, "config.json"),
        Path(DRAFT_MODEL_PATH, "config.json"),
        Path(MODEL_STORE_PATH, ".modal-model-manifest.json"),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("model Volume is incomplete: " + ", ".join(missing))
