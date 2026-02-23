# 🎙️ Speech-to-Text — Real-time WebSocket STT

A production-ready, low-latency speech-to-text backend built with **FastAPI**, **faster-whisper**, and **CUDA** (RTX 3060 Ti / 8 GB VRAM).

## Architecture

```
Client  ──ws──▶  /ws/stt  ──▶  asyncio.Queue  ──▶  Worker (thread)
                                                        │
                                                  faster-whisper
                                                   (GPU, fp16)
                                                        │
Client  ◀──ws──  JSON result  ◀────────────────────────┘
```

- Model loaded **once** at startup; shared across all connections.
- Transcription runs in `asyncio.to_thread` so the event loop stays free.
- Supports **multiple concurrent WebSocket clients**.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 |
| NVIDIA GPU driver | ≥ 525 |
| CUDA Toolkit | 12.x (comes with PyTorch wheel) |
| cuDNN | bundled via `nvidia-cudnn-cu12` |

---

## Setup

### 1. Create & activate a virtual environment

```powershell
cd "f:\LLM Project\speech-to-text"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install project dependencies

```powershell
pip install -r requirements.txt
```

### 4. (Optional) Verify CUDA is available

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True NVIDIA GeForce RTX 3060 Ti`

---

## Running the server

```powershell
python run.py
```

The server starts at `http://0.0.0.0:8000`. On first run, the Whisper model will be downloaded automatically (~500 MB for `small`).

### Health check

```powershell
curl http://localhost:8000/health
# → {"status":"ok","model":"small","device":"cuda"}
```

---

## WebSocket API

### Endpoint

```
ws://localhost:8000/ws/stt
```

### Protocol

1. **Connect** — you'll receive a `status` message:
   ```json
   {"type": "status", "message": "Connected. Send raw PCM 16 kHz mono 16-bit LE audio."}
   ```

2. **Send audio** — send binary frames of raw PCM audio.
   - Format: **16 kHz, mono, 16-bit signed little-endian** (no WAV header).
   - Recommended chunk size: **1–2 seconds** (32 000–64 000 bytes).

3. **Receive transcriptions** — JSON messages:
   ```json
   {
     "type": "partial",
     "text": "hello world",
     "confidence": 0.87,
     "language": "en",
     "processing_time_ms": 142.3
   }
   ```

---

## Quick Test (Python client)

Save as `test_client.py` and run while the server is active:

```python
import asyncio
import json
import numpy as np
import websockets

async def main():
    uri = "ws://localhost:8000/ws/stt"
    async with websockets.connect(uri) as ws:
        # Read status
        status = await ws.recv()
        print("Server:", json.loads(status))

        # Generate 1 second of silence (for smoke test)
        silence = np.zeros(16000, dtype=np.int16).tobytes()
        await ws.send(silence)

        # Wait for response
        response = await ws.recv()
        print("Result:", json.loads(response))

asyncio.run(main())
```

For a real test, send actual recorded PCM audio bytes instead of silence.

---

## Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `small` | Whisper model (`tiny`, `base`, `small`, `medium`) |
| `COMPUTE_TYPE` | `float16` | Precision (`float16`, `int8_float16`, `int8`) |
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `BEAM_SIZE` | `1` | 1 = greedy (fastest), higher = more accurate |
| `LANGUAGE` | `en` | ISO 639-1 code; empty string for auto-detect |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `MAX_QUEUE_SIZE` | `100` | Back-pressure limit |
| `NUM_WORKERS` | `1` | Number of queue workers |

Example with medium model:
```powershell
$env:MODEL_SIZE="medium"
python run.py
```

---

## Project Structure

```
speech-to-text/
├── app/
│   ├── __init__.py
│   ├── config.py          # Settings via pydantic-settings
│   ├── main.py            # FastAPI app + lifespan
│   ├── schemas.py         # JSON response models
│   ├── stt_service.py     # Whisper model + transcription
│   └── ws_endpoint.py     # WebSocket route + queue worker
├── requirements.txt
├── run.py                 # Uvicorn entrypoint
└── README.md
```