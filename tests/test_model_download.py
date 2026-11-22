"""Offline contract between modal-benchmark v0.2.0 and the Qwen3.8 GPU loader."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

from deploy.download_models import MANIFEST, main, sources, validate_model_store
from deploy.modal_config import CONFIG, UPSTREAM_LORA_IMAGE, build_sglang_command
from deploy import modal_app


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.record = {"version": 2, "downloader": "modal-benchmark"}
        self.record["target"] = {"repo_id": CONFIG.model_id, "revision": "a" * 40,
                                 "path": f"downloads/full/unit/{CONFIG.model_id}/{'a' * 40}/model"}
        self.record["draft"] = {"repo_id": CONFIG.draft_model_id, "revision": "b" * 40,
                                "path": f"downloads/full/unit/{CONFIG.draft_model_id}/{'b' * 40}/model"}
        for role in ("target", "draft"):
            folder = self.root / self.record[role]["path"]
            folder.mkdir(parents=True)
            (folder / "config.json").write_text("{}")
            (folder / "weights.safetensors").write_bytes(b"mock")
        self.save(self.record)

    def save(self, record):
        (self.root / MANIFEST).write_text(json.dumps(record), encoding="utf-8")

    def read(self, **options):
        return validate_model_store(
            str(self.root), CONFIG.model_id, CONFIG.draft_model_id, **options)

    def test_target_and_draft_paths(self):
        paths = self.read()
        for role in ("target", "draft"):
            self.assertEqual(paths[role], str((self.root / self.record[role]["path"]).resolve()))

    def test_sglang_uses_effective_paths(self):
        paths = self.read()
        args = build_sglang_command(target_path=paths["target"], draft_path=paths["draft"])
        self.assertEqual(args[args.index("--model-path") + 1], paths["target"])
        if CONFIG.speculative_algorithm != "NONE":
            self.assertEqual(args[args.index("--speculative-draft-model-path") + 1],
                             paths["draft"])
            self.assertEqual(args[args.index("--speculative-num-draft-tokens") + 1], "8")

    def test_no_draft_when_speculation_disabled(self):
        with patch("deploy.download_models.CONFIG",
                   replace(CONFIG, speculative_algorithm="NONE")):
            self.assertEqual(len(sources()), 1)
        self.record.pop("draft")
        self.save(self.record)
        self.assertEqual(set(self.read(require_draft=False)), {"target"})

    def test_immutable_revision_pin(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected model revision"):
            self.read(target_revision="c" * 40)

    def test_path_escape_rejected(self):
        self.record["target"]["path"] = "../outside"
        self.save(self.record)
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.read()

    def test_missing_shard_rejected(self):
        (self.root / self.record["draft"]["path"] / "weights.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "No model weight"):
            self.read()

    def test_legacy_manifest_rejected(self):
        self.save({"target": self.record["target"]})
        with self.assertRaisesRegex(RuntimeError, "v2 download manifest"):
            self.read()

    def test_reject_shard_index_directory_traversal(self):
        folder = self.root / self.record["target"]["path"]
        index = folder / "weights.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer": "../outside.safetensors"}}))
        (folder.parent / "outside.safetensors").write_bytes(b"fake")
        with self.assertRaisesRegex(RuntimeError, "unsafe indexed model shard"):
            self.read()

    def test_pi_lora_preflight_missing_config_and_weights(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            with (patch.object(modal_app, "PI_LORA_PATH", str(folder)),
                  patch.object(modal_app, "CONFIG", replace(CONFIG, pi_lora=True, sglang_image=UPSTREAM_LORA_IMAGE))):
                with self.assertRaisesRegex(RuntimeError, "adapter_config.json"):
                    modal_app._validate_pi_lora()
                (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "weights"):
                    modal_app._validate_pi_lora()
                (folder / "adapter_model.safetensors").write_bytes(b"mock")
                modal_app._validate_pi_lora()

    def test_pi_lora_disabled_skips_adapters(self):
        with patch.object(modal_app, "CONFIG", replace(CONFIG, pi_lora=False)):
            modal_app._validate_pi_lora()

    def test_single_download_call_publishes_both_roles(self):
        backend = Mock()
        backend.download.return_value = {
            "status": "VERIFIED", "models": [{"repo_id": CONFIG.model_id}],
            "end_to_end_seconds": 4.0, "timings": {"publish_seconds": 1},
        }
        with patch("modal_benchmark.hf.create_downloader", return_value=backend) as create:
            main(["--execute", "--workers", "2"])
        self.assertEqual(create.call_count, 1)
        self.assertEqual(backend.download.call_count, 1)
        self.assertEqual(backend.download.call_args.kwargs["active_manifest"], MANIFEST)
        self.assertEqual(backend.download.call_args.kwargs["aliases"], {
            "target": CONFIG.model_id, "draft": CONFIG.draft_model_id})

    def test_free_preview_does_not_create_downloader(self):
        with patch("modal_benchmark.hf.create_downloader") as create:
            main([])
        create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
