"""Local OpenAI-compatible streaming benchmark for the Modal server."""

from __future__ import annotations

import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier


@dataclass(frozen=True)
class StreamResult:
    request_id: int
    completion_tokens: int
    finish_reason: str
    total_seconds: float
    ttft_seconds: float
    decode_seconds: float
    decode_tps: float


def _run_stream(
    endpoint: str,
    model_name: str,
    max_tokens: int,
    request_id: int,
    start_barrier: Barrier,
    timeout_seconds: int,
) -> StreamResult:
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Request {request_id}: Write a long, dense and continuous technical "
                    "explanation of high-performance LLM inference. Cover GPU kernels, "
                    "memory bandwidth, KV caching, quantization, speculative decoding, "
                    "CUDA graphs and scheduling. Do not conclude early; continue until "
                    "the output token limit."
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

    start_barrier.wait()
    request_started = time.perf_counter()
    first_output_at: float | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    finished_at: float | None = None
    saw_done = False

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue

            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
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
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])

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

    if not saw_done:
        raise RuntimeError(
            f"request {request_id}: stream ended before the OpenAI SSE [DONE] marker"
        )
    if finished_at is None or first_output_at is None:
        raise RuntimeError(f"request {request_id}: completed without an output token")
    if completion_tokens is None:
        raise RuntimeError(
            f"request {request_id}: completed without usage.completion_tokens"
        )

    total_seconds = finished_at - request_started
    ttft_seconds = first_output_at - request_started
    decode_seconds = max(finished_at - first_output_at, 1e-9)
    decode_tps = max(completion_tokens - 1, 0) / decode_seconds
    return StreamResult(
        request_id=request_id,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason or "unknown",
        total_seconds=total_seconds,
        ttft_seconds=ttft_seconds,
        decode_seconds=decode_seconds,
        decode_tps=decode_tps,
    )


def run_concurrency_benchmark(
    base_url: str,
    model_name: str,
    max_tokens: int,
    concurrency: int,
    timeout_seconds: int = 3600,
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")

    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    barrier = Barrier(concurrency + 1)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _run_stream,
                endpoint,
                model_name,
                max_tokens,
                request_id,
                barrier,
                timeout_seconds,
            )
            for request_id in range(1, concurrency + 1)
        ]
        batch_started = time.perf_counter()
        barrier.wait()
        results = [future.result() for future in futures]
        batch_seconds = time.perf_counter() - batch_started

    total_tokens = sum(result.completion_tokens for result in results)
    aggregate_tps = total_tokens / max(batch_seconds, 1e-9)
    avg_decode_tps = statistics.mean(result.decode_tps for result in results)
    avg_ttft = statistics.mean(result.ttft_seconds for result in results)

    print()
    print("=" * 88)
    print(f"Qwen3.8-27B Modal benchmark | concurrency={concurrency}")
    print("=" * 88)
    print(f"Endpoint:             {base_url}")
    print(f"Tokens per request:   {max_tokens}")
    for result in results:
        print(
            f"req {result.request_id:>2}: tokens={result.completion_tokens:<5} "
            f"finish={result.finish_reason:<8} TTFT={result.ttft_seconds:>6.3f}s "
            f"decode={result.decode_tps:>7.2f} tok/s"
        )
    print("-" * 88)
    print(f"Batch wall time:      {batch_seconds:.3f} s")
    print(f"Total output tokens:  {total_tokens}")
    print(f"Avg user TTFT:        {avg_ttft:.3f} s")
    print(f"Avg user decode:      {avg_decode_tps:.2f} tok/s")
    print(f"Aggregate end-to-end: {aggregate_tps:.2f} tok/s")
    print("=" * 88)
    print("Every request completed with SSE [DONE] and usage.completion_tokens.")


def run_single_stream_benchmark(
    base_url: str,
    model_name: str,
    max_tokens: int,
    timeout_seconds: int = 3600,
) -> None:
    """Backward-compatible single-stream wrapper."""
    run_concurrency_benchmark(
        base_url,
        model_name,
        max_tokens,
        concurrency=1,
        timeout_seconds=timeout_seconds,
    )
