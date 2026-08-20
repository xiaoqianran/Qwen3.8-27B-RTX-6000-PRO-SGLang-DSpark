# DFlash 2 backport overlay for `lmsysorg/sglang:qwen38-27b`

This directory is a **read-only, git-tracked overlay** that adds upstream SGLang's
DFlash 2 support to the cookbook-pinned image `lmsysorg/sglang:qwen38-27b`
(`0.0.0.dev0+qwen38.27b.g561c8f3`), which predates DFlash 2 and only ships the
DFlash 1 model class.

`start.sh` bind-mounts the individual files here into
`/sgl-workspace/sglang/python/sglang/...` (the image's editable source tree) at
launch and refuses to start if any file is missing.

## Why this is needed

DFlash 2 (block-diffusion drafter with two-tap dynamic convolutions and a
candidate-path selector) landed in upstream SGLang only on **2026-08-19**:

- `sgl-project/sglang` PR #35371 "DFlash2: local convolution + candidate selector"
- `sgl-project/sglang` PR #35496 "Support quantized target lm_head in the DFlash2 selector"

The cookbook image predates both, so `z-lab/Qwen3.8-27B-DFlash2` (config
architecture `DFlash2DraftModel`) fails on the stock image with:

```
ValueError: Cannot find model module. 'DFlash2DraftModel' is not a registered model
in the Transformers library ... and 'AutoModel' is not present in the model config's
'auto_map' ...
```

## What is patched

Files replaced wholesale with upstream `main` (DFlash 2 feature-set only; DSPARK
workers are untouched):

| Patched file | Upstream change |
|---|---|
| `srt/models/dflash.py` | adds `DFlash2DraftModel`, `CandidateSelector`, `DFlashGroupedConv` |
| `kernels/ops/speculative/dflash.py` | adds `selector_walk_triton` |
| `srt/speculative/dflash_utils.py` | adds `is_dense_head_weight`, `table_qk_norm_rope_` |
| `srt/speculative/dflash_worker_v2.py` | DFlash 2 worker |
| `srt/speculative/dflash_info.py` | DFlash 2 verify input |
| `srt/speculative/dflash_info_v2.py` | DFlash 2 draft input |
| `srt/speculative/draft_worker_common.py` | DFlash 2 draft worker plumbing |

Files shared with other spec paths (DSPARK/EAGLE/MTP) get **appended-onto only**
(i.e. the image's file + the specific upstream function, nothing removed), so the
other algorithms keep working:

| Appended-to file | Added function |
|---|---|
| `srt/speculative/spec_utils.py` | `sample_simulated_acc_len` |
| `srt/mem_cache/allocation_sizing.py` | `page_aligned_decode_alloc_lens` |
| `srt/layers/moe/utils.py` | `draft_model_build_scope` (adapted to this image's `speculative_context` flag) |
| `srt/layers/logprob_processor.py` | `compute_spec_logprobs` |

## How to regenerate (update to a newer upstream)

```bash
# from repo root; upstream files written under ./patch/_fetch
python3 patch/sglang/gen_patch.sh   # or copy the corresponding
                                    # sgl-project/sglang main files by hand
```

The overlay is meant to be swapped out for the real upstream SGLang once a docker
image with DFlash 2 ships (then set `IMAGE`/delete `PATCH_DIR` mount and run a
stock `launch_server` with `--speculative-algorithm DFLASH`). Until then, this
overlay is the minimal path to DFlash 2 on the cookbook image.
