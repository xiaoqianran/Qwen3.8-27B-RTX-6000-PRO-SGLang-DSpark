# Modal autoscaling behavior

Clients call `Qwen38HTTP`, the RTX PRO 6000 SGLang Server in the
`qwen38-27b-modal` App. It uses `min_containers=0`, `max_containers=1`, and a
600-second scaledown window.

## GPU idle window

Modal starts a GPU container when a request reaches a Server with no ready replica.
A browser or Cherry Studio conversation being open does not count as activity.
An in-flight generation keeps demand active. When the last request finishes, the
GPU may stay warm for up to 600 seconds. Override this window with
`QWEN38_GPU_SCALEDOWN_SECONDS`.

All HTTP requests to the Server, including model discovery and health checks,
can create or refresh GPU demand. If you need the GPU to scale to zero, turn
off periodic checks against the Server URL.

## Cold starts

The Modal web endpoint waits for the GPU container to become ready. Set an appropriate client timeout for cold starts. Cold
start duration depends on the model, image and cache state.

## Deployment and shutdown

Deploy `deploy/modal_app.py` to update `Qwen38HTTP`. A redeploy can roll the
GPU revision. The shutdown hook terminates SGLang when the container exits.

When migrating from the older CPU gateway deployment, switch clients to the
direct Server URL before retiring the old App. See [deployment instructions](./README.md).
