"""Add Bearer authentication inside the SGLang HTTP process."""

import hmac
import os
import runpy


class BearerAuth:
    def __init__(self, app):
        self.app = app
        self.key = os.environ["SGLANG_HTTP_API_KEY"].encode()
        if not self.key:
            raise RuntimeError("SGLANG_HTTP_API_KEY cannot be empty")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        value = dict(scope["headers"]).get(b"authorization", b"")
        scheme, _, key = value.partition(b" ")
        if scheme.lower() == b"bearer" and hmac.compare_digest(self.key, key):
            await self.app(scope, receive, send)
            return
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
        await send({"type": "http.response.body", "body": b'{"error":"Unauthorized"}'})


if __name__ == "__main__":
    from sglang.srt.entrypoints.http_server import app

    app.add_middleware(BearerAuth)
    runpy.run_module("sglang.launch_server", run_name="__main__")
