"""JSON response schemas for the WebSocket API."""

from __future__ import annotations

from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    """Partial transcription pushed back to the client."""

    type: str = "partial"
    text: str
    confidence: float
    language: str = ""
    processing_time_ms: float = 0.0


class ErrorResult(BaseModel):
    """Error message pushed to the client."""

    type: str = "error"
    message: str


class StatusResult(BaseModel):
    """Connection status message."""

    type: str = "status"
    message: str
