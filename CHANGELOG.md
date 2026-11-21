# Changelog

All notable changes to this project are documented here. Dates are commit dates.

## 2026-08-25 — Default NVFP4 uses the dense BF16 `lm_head`

SGLang now ships two NVFP4 checkpoints ([cookbook split](https://github.com/sgl-project/sglang/pull/36020)): packed-FP4 `lm_head` vs dense BF16. Cookbook recipes (including RTX PRO 6000) were measured against the BF16-head export, so this repo follows that.

**Changed:**

- Default `MODEL_ID` is now `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` instead of `RadixArk/Qwen3.8-27B-NVFP4`. Override with `MODEL_ID=… ./start.sh` (packed-FP4 twin, full BF16, or FP8).
- **Existing users:** Hugging Face caches per repo — after `git pull`, delete `.cache/huggingface/hub/models--RadixArk--Qwen3.8-27B-NVFP4` (unless you still set `MODEL_ID` to that repo) or the next start will download ~24 GB *and* leave the old ~22 GB on disk. The DFlash2 draft cache is unchanged.
- Size vs the packed-FP4 twin (`248320 × 5120` `lm_head` only; body/draft/KV/GDN unchanged): **~+1.7 GB on disk**. Packed NVFP4 head is ~0.67 GiB; dense BF16 is `248320×5120×2` ≈ **2.37 GiB**, so the weight delta is ~1.7 GiB. The cookbook’s “~3.2 GB larger at runtime” is the dense head’s **own** working set (weights + allocator/CUDA-graph slack), not +3.2 GB on top of the old packed head. Expect ~+1.7 GB extra head weights vs before; resident head can look ~2.4–3.2 GB. `--mem-fraction-static 0.90` is unchanged (that pin was measured on the BF16-head export).
