"""Local OpenAI-compatible streaming benchmark for the Modal server."""

from __future__ import annotations

import json
import time
import urllib.request


def run_single_stream_benchmark(
    base_url: str,
    model_name: str,
    max_tokens: int,
    timeout_seconds: int = 1800,
) -> None:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a long, dense and continuous technical explanation of "
                    "high-performance LLM inference. Cover GPU kernels, memory "
                    "bandwidth, KV caching, quantization, speculative decoding, "
                    "CUDA graphs and scheduling. Do not conclude early; continue "
                    "until the output token limit."
                ),
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    request_started = time.perf_counter()
    first_output_at: float | None = None
    completion_tokens: int | None = None
    finished_at: float | None = None

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                finished_at = time.perf_counter()
                break
            if not data:
                continue

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            usage = event.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                completion_tokens = int(usage["completion_tokens"])

            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                emitted = (
                    isinstance(content, str) and bool(content)
                ) or (
                    isinstance(reasoning, str) and bool(reasoning)
                )
                if emitted and first_output_at is None:
                    first_output_at = time.perf_counter()

    if finished_at is None:
        finished_at = time.perf_counter()

    total_seconds = finished_at - request_started

    print()
    print("=" * 76)
    print("Qwen3.8-27B Modal single-stream decode benchmark")
    print("=" * 76)
    print(f"Endpoint:          {base_url}")
    print(f"Requested tokens:  {max_tokens}")
    print(f"Total time:        {total_seconds:.3f} s")

    if first_output_at is None:
        print("No streamed output token observed.")
        print("=" * 76)
        return

    ttft = first_output_at - request_started
    decode_seconds = max(finished_at - first_output_at, 1e-9)
    print(f"Observed TTFT:     {ttft:.3f} s")
    print(f"Decode window:     {decode_seconds:.3f} s")

    if completion_tokens is None:
        print("Completion tokens: unavailable from stream usage")
        print("Decode tok/s:      unavailable")
    else:
        decode_tokens = max(completion_tokens - 1, 0)
        decode_tps = decode_tokens / decode_seconds
        end_to_end_tps = completion_tokens / total_seconds
        print(f"Completion tokens: {completion_tokens}")
        print(f"DECODE TOK/S:      {decode_tps:.2f}")
        print(f"End-to-end tok/s:  {end_to_end_tps:.2f}")

    print("=" * 76)
    print("Cross-check with SGLang's decode throughput in the Modal server logs.")
