# Modal autoscaling behavior

This deployment has two independent scale-to-zero layers:

- `Qwen38Gateway`: lightweight ASGI Web Function, 0.125 CPU, `min_containers=0`, 5-second scaledown window.
- `Qwen38Backend`: RTX PRO 6000 SGLang Server, `min_containers=0`, `max_containers=1`, 600-second scaledown window.

## Why the GPU window is explicit

Modal's `scaledown_window` is the maximum amount of time an idle container waits before autoscaling down. A browser or Cherry Studio conversation being open does **not** count as activity; only an in-flight HTTP request does.

Before the GPU window was pinned, the Server used Modal's default idle behavior. A production log sample on 2026-08-21 showed the last `/v1/chat/completions` request at `08:53:41 UTC`, followed by `SIGTERM` at `08:54:49 UTC` with `Remaining number of requests 0`. That approximately 68-second gap is consistent with the prior short default idle window and explains why the GPU disappeared between chat turns.

The explicit 600-second window makes the intended behavior deterministic for interactive chat:

1. An in-flight generation keeps the GPU request active.
2. When the final request finishes, the idle window begins.
3. Any new real backend request within the next 600 seconds refreshes demand.
4. If no backend request arrives for roughly 10 minutes, Modal may scale the GPU to zero.

`exit_grace_period` is separate. It protects in-flight requests after a shutdown has already been initiated; it is not an idle keep-alive timer.

## Cold-start interaction

The GPU cold start is roughly 90-120 seconds on the validated profile. A 600-second GPU scaledown window is intentionally longer than cold start, preventing a scale-from-zero trigger from becoming idle and being torn down immediately after the model finally becomes ready.

The CPU gateway has a much shorter 5-second idle window because it is cheap to restart and must not incur a persistent idle charge. A live streaming request keeps the gateway invocation active; the 5-second window applies only after requests finish.

## Deployment object names

The legacy deployment used the Modal object name `Qwen38Server` for an `@app.server`. Modal does not allow an existing object to change in place from Server to Function. The scale-to-zero ASGI gateway therefore uses the Modal object name `Qwen38Gateway` while keeping the public ASGI endpoint label `qwen38server`.
