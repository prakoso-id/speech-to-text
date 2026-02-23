"""
Speech-to-text service: model lifecycle + transcription helper.

The WhisperModel is loaded once at startup and shared across all
connections.  Transcription is CPU/GPU-bound, so it is dispatched
via ``asyncio.to_thread`` to keep the event loop responsive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from io import BytesIO
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

from app.config import settings

logger = logging.getLogger(__name__)

# ── Module-level singleton ──────────────────────────────────────────
_model: WhisperModel | None = None


def load_model() -> WhisperModel:
    """
    Instantiate and cache the WhisperModel.

    Called once during application startup.
    """
    global _model
    if _model is not None:
        logger.info("Model already loaded – reusing cached instance.")
        return _model

    logger.info(
        "Loading faster-whisper model  size=%s  device=%s  compute=%s …",
        settings.model_size,
        settings.device,
        settings.compute_type,
    )
    start = time.perf_counter()
    _model = WhisperModel(
        settings.model_size,
        device=settings.device,
        compute_type=settings.compute_type,
    )
    elapsed = time.perf_counter() - start
    logger.info("Model loaded in %.2f s", elapsed)
    return _model


def warmup_model() -> None:
    """
    Run a dummy transcription on 1 s of silence to pre-heat CUDA kernels.

    This eliminates the cold-start latency on the first real request.
    """
    logger.info("Warming up CUDA pipeline …")
    start = time.perf_counter()
    dummy_audio = np.zeros(settings.sample_rate, dtype=np.float32)
    model = get_model()
    segments, _ = model.transcribe(dummy_audio, beam_size=1)
    # Materialise the generator to force computation.
    for _ in segments:
        pass
    elapsed = time.perf_counter() - start
    logger.info("Warm-up complete in %.2f s", elapsed)


def get_model() -> WhisperModel:
    """Return the cached model or raise if not loaded."""
    if _model is None:
        raise RuntimeError("WhisperModel has not been loaded yet.")
    return _model


# ── Audio conversion ────────────────────────────────────────────────

def pcm_bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
    """
    Convert raw PCM 16-bit signed LE mono audio to float32 in [-1, 1].

    Parameters
    ----------
    audio_bytes : bytes
        Raw PCM data (16 kHz, mono, 16-bit signed little-endian).

    Returns
    -------
    np.ndarray
        Audio waveform as float32 values normalised to [-1.0, 1.0].
    """
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    return audio_float32


# ── Transcription ───────────────────────────────────────────────────

def _transcribe_sync(audio_float32: np.ndarray) -> dict[str, Any]:
    """
    Run transcription synchronously (called inside a thread).

    Returns a dict ready to be serialised as JSON.
    """
    model = get_model()
    start = time.perf_counter()

    kwargs: dict[str, Any] = {
        "beam_size": settings.beam_size,
        "vad_filter": True,
        "vad_parameters": {
            "min_silence_duration_ms": 200,
        },
    }
    if settings.language:
        kwargs["language"] = settings.language

    segments, info = model.transcribe(audio_float32, **kwargs)

    # Materialise segments (generator) into a single string.
    texts: list[str] = []
    total_logprob = 0.0
    seg_count = 0
    for seg in segments:
        texts.append(seg.text.strip())
        total_logprob += seg.avg_logprob
        seg_count += 1

    elapsed_ms = (time.perf_counter() - start) * 1000
    text = " ".join(texts).strip()
    avg_confidence = (
        round(2 ** (total_logprob / seg_count), 4) if seg_count else 0.0
    )

    return {
        "type": "partial",
        "text": text,
        "confidence": avg_confidence,
        "language": info.language,
        "processing_time_ms": round(elapsed_ms, 1),
    }


async def transcribe_audio(audio_bytes: bytes) -> dict[str, Any]:
    """
    Non-blocking transcription entry-point.

    Converts raw PCM to float32 and runs faster-whisper in a thread
    so the asyncio event loop stays free for WebSocket I/O.
    """
    audio_float32 = pcm_bytes_to_float32(audio_bytes)

    # Reject very short / empty chunks early.
    duration_s = len(audio_float32) / settings.sample_rate
    if duration_s < 0.1:
        return {
            "type": "partial",
            "text": "",
            "confidence": 0.0,
            "language": settings.language,
            "processing_time_ms": 0.0,
        }

    result = await asyncio.to_thread(_transcribe_sync, audio_float32)
    return result
