import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from deploy.download_models import published_record, validate_model_store, check_completion_receipt, sources
from deploy.modal_config import CONFIG, build_sglang_command

class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.manifest = {"version": 1, "models": [
            {"repo_id": CONFIG.model_id, "revision": "a"*40, "files": {}},
            {"repo_id": CONFIG.draft_model_id, "revision": "b"*40, "files": {}},
        ]}
        self.record = published_record(self.manifest, "/models/downloads/full/unit")
        for role in ("target", "draft"):
            path = self.root / self.record[role]["path"]
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").write_bytes(b"mock")
        self.save(self.record)

    def save(self, record):
        (self.root / ".modal-model-manifest.json").write_text(json.dumps(record))

    def read(self, **kw):
        return validate_model_store(str(self.root), CONFIG.model_id,
                                    CONFIG.draft_model_id, **kw)

    def test_benchmark_receipt_location(self):
        # modal_benchmark.hf.download.repo_root(base, model) is the
        # parent of the actual downloaded weights folder ("model").
        from modal_benchmark.hf.download import repo_root

        base = "/models/downloads/full/unit"
        repo = repo_root(Path(base), self.manifest["models"][0])
        model = Path("/models") / self.record["target"]["path"]
        self.assertEqual(repo / "model", model)
        self.assertEqual(repo / "all-components-complete.json",
                         model.parent / "all-components-complete.json")
        self.assertNotEqual(repo / "all-components-complete.json",
                            model / "all-components-complete.json")

    def test_completion_receipt_validation(self):
        folder = self.root / self.record["target"]["path"]
        marker = folder.parent / "all-components-complete.json"
        with self.assertRaisesRegex(RuntimeError, "completion marker missing"):
            check_completion_receipt(self.manifest, self.record, self.root)

        marker.write_text(json.dumps({
            "status": "COMPLETE",
            "manifest": self.manifest,
            "verification_mode": "xet",
        }))
        check_completion_receipt(self.manifest, self.record, self.root)
        marker.write_text(json.dumps({
            "status": "COMPLETE",
            "manifest": self.manifest,
            "verification_mode": "fast",
        }))
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            check_completion_receipt(self.manifest, self.record, self.root)

    def test_no_draft_download_without_dflash(self):
        with patch("deploy.download_models.CONFIG",
                   replace(CONFIG, speculative_algorithm="NONE")):
            self.assertEqual(len(sources()), 1)
            record = published_record(
                {"version": 1, "models": [self.manifest["models"][0]]},
                "/models/downloads/full/unit",
            )
        self.assertNotIn("draft", record)
        self.save(record)
        self.assertEqual(set(self.read(require_draft=False)), {"target"})

    def test_publisher_does_not_infer_draft_from_remote_env(self):
        single = {"version": 1, "models": [self.manifest["models"][0]]}
        # The local scheduler may have speculative mode NONE while the
        # publisher's worker has only default environment variables.
        with patch("deploy.download_models.CONFIG",
                   replace(CONFIG, speculative_algorithm="DFLASH")):
            record = published_record(single, "/models/downloads/full/unit")
        self.assertEqual(set(record) - {"version", "downloader"}, {"target"})

    def test_paths_and_command(self):
        paths = self.read()
        self.assertEqual(paths["target"], str((self.root / self.record["target"]["path"]).resolve()))
        args = build_sglang_command(target_path=paths["target"], draft_path=paths["draft"])
        self.assertEqual(args[args.index("--model-path")+1], paths["target"])
        if CONFIG.speculative_algorithm != "NONE":
            self.assertEqual(args[args.index("--speculative-draft-model-path")+1], paths["draft"])

    def test_reject_escape(self):
        self.record["target"]["path"] = "../escape"
        self.save(self.record)
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.read()

    def test_reject_missing_weights(self):
        (self.root / self.record["draft"]["path"] / "model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "No model weight"):
            self.read()

    def test_reject_pinned_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected model revision"):
            self.read(target_revision="c"*40)

    def test_reject_legacy(self):
        self.save({"target": {"repo_id": CONFIG.model_id}})
        with self.assertRaisesRegex(RuntimeError, "v2 download manifest"):
            self.read()

if __name__ == "__main__":
    unittest.main()
