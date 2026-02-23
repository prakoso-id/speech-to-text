"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration.

    All values can be overridden with environment variables
    (case-insensitive, e.g. MODEL_SIZE=medium).
    """

    # ── Whisper model ───────────────────────────────────────────────
    model_size: str = "small"          # "tiny", "base", "small", "medium", "large-v3"
    compute_type: str = "float16"      # "float16", "int8_float16", "int8"
    device: str = "cuda"               # "cuda" or "cpu"
    beam_size: int = 1                 # 1 = greedy (fastest); increase for accuracy
    language: str = "en"               # ISO-639-1 code; set to "" for auto-detect

    # ── Server ──────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Audio ───────────────────────────────────────────────────────
    sample_rate: int = 16000           # Expected PCM sample rate
    sample_width: int = 2              # 16-bit = 2 bytes per sample

    # ── Queue / concurrency ─────────────────────────────────────────
    max_queue_size: int = 100          # Back-pressure: drop if queue is full
    num_workers: int = 1               # GPU workers (1 is optimal for single GPU)

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()
