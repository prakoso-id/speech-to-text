"""
WebSocket endpoint ``/ws/stt`` and the background queue worker.

Architecture
------------
1. Each connected client sends raw PCM audio frames as binary
   WebSocket messages.
2. Frames are placed on a shared ``asyncio.Queue`` as
   ``(WebSocket, bytes)`` tuples.
3. A long-lived background worker pulls items from the queue,
   transcribes them via ``stt_service``, and pushes the JSON
   result back to the originating WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.config import settings
from app.schemas import ErrorResult, StatusResult, TranscriptionResult
from app.stt_service import transcribe_audio

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Shared state ────────────────────────────────────────────────────
audio_queue: asyncio.Queue[tuple[WebSocket, bytes]] = asyncio.Queue(
    maxsize=settings.max_queue_size,
)
_active_connections: Set[WebSocket] = set()


# ── Queue worker ────────────────────────────────────────────────────

async def queue_worker() -> None:
    """
    Consume ``(ws, audio_bytes)`` items from the queue forever.

    Errors on individual items are caught so one bad chunk never
    crashes the worker.
    """
    logger.info("Queue worker started (num_workers=%d)", settings.num_workers)
    while True:
        ws, audio_bytes = await audio_queue.get()
        try:
            # Skip if the client already disconnected.
            if ws not in _active_connections:
                logger.debug("Skipping chunk – client already disconnected.")
                continue

            result = await transcribe_audio(audio_bytes)

            # Only send if there is actual text or if result is meaningful
            if ws in _active_connections:
                await ws.send_json(result)

        except WebSocketDisconnect:
            logger.info("Client disconnected while sending result.")
            _active_connections.discard(ws)
        except Exception:
            logger.exception("Error processing audio chunk")
            try:
                if ws in _active_connections:
                    err = ErrorResult(message="Transcription failed for this chunk.")
                    await ws.send_json(err.model_dump())
            except Exception:
                logger.debug("Could not send error to client (likely disconnected).")
                _active_connections.discard(ws)
        finally:
            audio_queue.task_done()


# ── WebSocket endpoint ──────────────────────────────────────────────

@router.websocket("/ws/stt")
async def websocket_stt(ws: WebSocket) -> None:
    """
    Accept a WebSocket connection, receive audio chunks, and
    enqueue them for transcription.
    """
    await ws.accept()
    _active_connections.add(ws)
    client_id = id(ws)
    logger.info("Client %s connected  (total=%d)", client_id, len(_active_connections))

    # Let the client know we're ready.
    status = StatusResult(message="Connected. Send raw PCM 16 kHz mono 16-bit LE audio.")
    await ws.send_json(status.model_dump())

    try:
        while True:
            data = await ws.receive_bytes()

            if audio_queue.full():
                logger.warning(
                    "Queue full – dropping chunk from client %s", client_id,
                )
                err = ErrorResult(message="Server is overloaded. Audio chunk dropped.")
                await ws.send_json(err.model_dump())
                continue

            await audio_queue.put((ws, data))

    except WebSocketDisconnect:
        logger.info("Client %s disconnected gracefully.", client_id)
    except Exception:
        logger.exception("Unexpected error on client %s", client_id)
    finally:
        _active_connections.discard(ws)
        logger.info(
            "Client %s removed  (remaining=%d)", client_id, len(_active_connections),
        )
