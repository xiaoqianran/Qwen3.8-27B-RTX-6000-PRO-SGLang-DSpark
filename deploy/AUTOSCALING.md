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


## User-visible GPU wake policy

The public gateway follows a strict cost boundary:

| Request | Wakes a sleeping GPU? | Extends a hot GPU? |
| --- | --- | --- |
| `GET /_gateway/health` | No | No |
| `GET /health` | No | No |
| `GET /v1/models` | No | No |
| `POST /v1/chat/completions` | **Yes** | **Yes** |
| unknown/read-only routes | No | No |

GPU lifecycle state is stored in the lightweight Modal Dict
`qwen38-27b-runtime-state`. `Qwen38Backend` writes `starting`, then `ready`, and
refreshes a heartbeat every 10 seconds while alive. Graceful scale-down writes
`idle`. If a container disappears without its exit hook, a heartbeat older than
35 seconds is automatically treated as `idle`, so `/v1/models` cannot advertise
a dead GPU forever.

This means Cherry Studio may refresh its provider/model list as often as it
likes without allocating an RTX PRO 6000. The cost boundary is the user's real
generation action, not discovery/UI background traffic.

## Independent deploy boundary

The public CPU gateway and RTX PRO 6000 backend are deployed as **separate
Modal Apps**:

- backend: `qwen38-27b-modal` from `deploy/modal_app.py`
- gateway: `qwen38-27b-gateway` from `deploy/modal_gateway.py`

This is a reliability boundary, not just code organization. A `modal deploy` of
an App creates a new deployment version. When the Gateway and GPU Server lived
in the same App, a Gateway-only code change could roll the GPU Server revision.
A production incident on 2026-08-21 demonstrated exactly that: deployment v2
was created at 09:21 UTC, a new RTX PRO 6000 revision began scheduling at
09:22:56, and the old GPU received SIGTERM at 09:23:23 immediately after its
183-second request completed. The later 09:25:11 App stop was a separate manual
Dashboard action.

With the split Apps, routine Gateway updates cannot restart or replace a hot GPU.
Only deploying `deploy/modal_app.py` can roll the RTX PRO 6000 backend. Never
deploy the backend while interactive inference is in progress unless that restart
is intentional.
