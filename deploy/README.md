# Modal deployment

`deploy/` is the Modal-native lifecycle for this repository. It does not execute
`start.sh` or `stop.sh`; it reuses the existing `patch/sglang` DFlash2 backport
so the Docker and Modal backends do not maintain duplicate compatibility code.

## Architecture

```text
CPU preparation (8 CPU, no GPU)
        │
        ├── RadixArk/Qwen3.8-27B-NVFP4
        └── z-lab/Qwen3.8-27B-DFlash2
        ▼
qwen38-27b-model-store (persistent Volume)
        │ read-only
        ▼
lmsysorg/sglang:qwen38-27b + patch/sglang
        ▼
max 1 × RTX PRO 6000
        ▼
SGLang + DFlash2 + FlashInfer
        └── max 8 active requests
```

The GPU runtime is offline from Hugging Face and uses only:

```text
--model-path /models/target
--speculative-draft-model-path /models/draft
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

CPU preparation verifies model metadata and all safetensors shards before GPU
allocation. If model revisions are not pinned, the CPU preparation step is
re-evaluated on each deploy; `snapshot_download` still reuses the persistent
Volume and only fetches missing/changed data.

## SGLang runtime choice

The Qwen3.8 cookbook image predates DFlash2. The Modal image therefore bakes the
repository's existing DFlash2 backport into `lmsysorg/sglang:qwen38-27b` and
fails during image build unless all of these integration points import:

```text
DFlash2DraftModel
DFlashWorkerV2
FusedKVMaterializeHelper
should_apply_lm_head_quant_method
```

It also verifies that the base server exposes `--max-mamba-cache-size`. No RTX
PRO 6000 is allocated for these checks.

## RTX PRO 6000 profile

The defaults follow the current SGLang RTX PRO 6000 + NVFP4 baseline and then
add DFlash2:

```text
GPU                         RTX-PRO-6000
SGLang image                lmsysorg/sglang:qwen38-27b
Target                      RadixArk/Qwen3.8-27B-NVFP4
Draft                       z-lab/Qwen3.8-27B-DFlash2
Attention                   flashinfer
KV cache                    fp8_e4m3
Context                     262144
mem-fraction-static         0.85
chunked-prefill-size        2048
DFlash2 draft tokens        8
SGLang max running requests 8
max-mamba-cache-size        40
Modal max containers        1
```

Qwen3.8 is a hybrid-GDN model. With the default `extra_buffer` strategy and the
overlap scheduler, the target needs five base recurrent-state slots per active
request. `8 × 5 = 40`, so the explicit Mamba-cache pin prevents the state pool
from silently reducing the requested eight-way SGLang concurrency. DFlash2's
draft model is pure attention, but speculative verification still uses the
target model's recurrent state and therefore does not remove this requirement.

`--min-free-slots-delay` is intentionally not overridden, and the RTX PRO 6000
cookbook's 2048 prefill chunk is used instead of the older fork's 4096 value.

## One-time Modal workspace setup

Use Image Builder `2025.06`:

```bash
uv run modal workspace settings set image-builder-version 2025.06
uv run modal workspace settings list
```

The image also restores `typing_extensions==4.16.0` as a compatibility guard.

## Run

Temporary benchmark:

```bash
uv run modal run deploy/modal_app.py --max-tokens 2048
```

Persistent endpoint:

```bash
uv run modal deploy deploy/modal_app.py
```

Runtime limits are deliberately simple:

```text
min_containers   0
max_containers   1
GPU replicas     0 or 1
SGLang requests  up to 8 inside that one GPU
```

The local benchmark retries expected 502/503/504 responses while the Modal
Server scales from zero. The endpoint is currently `unauthenticated=True`; add
authentication before exposing a paid production endpoint publicly.