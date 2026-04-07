#!/usr/bin/env python3
"""
NIXL PD Disaggregated Proxy for vLLM.

Routes requests: prefill (kv_producer) → decode (kv_consumer)
with proper kv_transfer_params forwarding for NIXL cross-node KV transfer.

Usage:
    python3 nixl_proxy.py \
        --model nvidia/Kimi-K2.5-NVFP4 \
        --prefill 172.31.53.87:8010 \
        --decode 172.31.50.97:8020 \
        --port 8000
"""

import argparse
import itertools
import json
import logging
import os
import sys
import uuid

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger("nixl_proxy")
logging.basicConfig(level=logging.INFO)

app = FastAPI()


class NixlProxy:
    def __init__(self, prefill_instances, decode_instances, model):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model

    def next_prefill(self):
        return next(self.prefill_cycler)

    def next_decode(self):
        return next(self.decode_cycler)


proxy: NixlProxy = None


@app.get("/v1/models")
async def list_models():
    return JSONResponse({"object": "list", "data": [{"id": proxy.model, "object": "model"}]})


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    return await _handle_request(raw_request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(raw_request: Request):
    return await _handle_request(raw_request, "/v1/completions")


async def _handle_request(raw_request: Request, endpoint: str):
    request_data = await raw_request.json()
    request_id = str(uuid.uuid4())

    prefill_host = proxy.next_prefill()
    decode_host = proxy.next_decode()

    # --- Phase 1: Prefill ---
    # Send to prefill with do_remote_decode=True, max_tokens=1, stream=False
    prefill_req = request_data.copy()
    prefill_req["max_tokens"] = 1
    if "max_completion_tokens" in prefill_req:
        prefill_req["max_completion_tokens"] = 1
    prefill_req["stream"] = False
    if "stream_options" in prefill_req:
        del prefill_req["stream_options"]
    prefill_req["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
    }

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "X-Request-Id": request_id,
    }

    kv_transfer_params = None
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(
                f"http://{prefill_host}{endpoint}",
                json=prefill_req,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error("Prefill failed (%s): %s", resp.status, error)
                    return JSONResponse(
                        status_code=resp.status,
                        content={"error": f"Prefill failed: {error}"},
                    )
                prefill_response = await resp.json()
                kv_transfer_params = prefill_response.get("kv_transfer_params")
                logger.info(
                    "Prefill done [%s], kv_transfer_params=%s",
                    request_id[:8],
                    kv_transfer_params,
                )
    except Exception as e:
        logger.error("Prefill error: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})

    if not kv_transfer_params:
        logger.warning(
            "No kv_transfer_params from prefill [%s], decode will do its own prefill",
            request_id[:8],
        )

    # --- Phase 2: Decode ---
    # Forward original request to decode with kv_transfer_params from prefill
    decode_req = request_data.copy()
    if kv_transfer_params:
        decode_req["kv_transfer_params"] = kv_transfer_params

    is_stream = request_data.get("stream", False)

    try:
        if is_stream:
            return StreamingResponse(
                content=_stream_decode(decode_host, endpoint, decode_req, headers),
                media_type="text/event-stream",
            )
        else:
            async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
                async with session.post(
                    f"http://{decode_host}{endpoint}",
                    json=decode_req,
                    headers=headers,
                ) as resp:
                    body = await resp.json()
                    return JSONResponse(status_code=resp.status, content=body)
    except Exception as e:
        logger.error("Decode error: %s", e)
        return JSONResponse(status_code=502, content={"error": str(e)})


async def _stream_decode(host, endpoint, data, headers):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        async with session.post(
            f"http://{host}{endpoint}",
            json=data,
            headers=headers,
        ) as resp:
            async for chunk in resp.content.iter_chunked(1024):
                yield chunk


def main():
    parser = argparse.ArgumentParser(description="NIXL PD Disaggregated Proxy")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--prefill", nargs="+", required=True, help="host:port")
    parser.add_argument("--decode", nargs="+", required=True, help="host:port")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    global proxy
    proxy = NixlProxy(args.prefill, args.decode, args.model)

    logger.info("NIXL Proxy starting on %s:%d", args.host, args.port)
    logger.info("  Prefill: %s", args.prefill)
    logger.info("  Decode:  %s", args.decode)
    logger.info("  Model:   %s", args.model)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
