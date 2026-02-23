"""
FastAPI application entry-point.

* Loads the Whisper model once at startup.
* Starts the background queue-worker task(s).
* Exposes the WebSocket router and a health-check endpoint.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.stt_service import load_model, warmup_model
from app.ws_endpoint import queue_worker, router as ws_router

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""

    # 1. Load the Whisper model (blocks briefly – acceptable at boot).
    logger.info("=== Starting up ===")
    load_model()

    # 2. Warm up CUDA kernels with a dummy transcription.
    warmup_model()

    # 3. Spin up background worker(s).
    workers: list[asyncio.Task] = []
    for i in range(settings.num_workers):
        task = asyncio.create_task(queue_worker(), name=f"stt-worker-{i}")
        workers.append(task)
    logger.info("Launched %d queue worker(s).", len(workers))

    yield  # ← application is running

    # 3. Shutdown: cancel workers.
    logger.info("=== Shutting down ===")
    for task in workers:
        task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    logger.info("All workers stopped.")


# ── App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Speech-to-Text API",
    description="Real-time speech-to-text via WebSocket using faster-whisper.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


# ── Health check ────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.model_size, "device": settings.device}
