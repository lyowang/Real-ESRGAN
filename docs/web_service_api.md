# Real-ESRGAN Web Service API Documentation

A Flask-based HTTP API that wraps Real-ESRGAN's super-resolution inference, allowing remote clients to upscale images and videos via simple HTTP requests.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
pip install -r requirements_web.txt
python setup.py develop
```

### Start the Server

```bash
# Basic (auto-detect GPU/CPU)
python web_service.py

# Specify port and GPU
python web_service.py --port 8288 --gpu-id 0

# Force fp32 precision (use if fp16 causes issues)
python web_service.py --fp32

# Preload a model at startup (avoids cold-start on first request)
python web_service.py --preload RealESRGAN_x4plus
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `8288` | Port to listen on |
| `-g`, `--gpu-id` | auto | GPU device index (0, 1, 2...) |
| `--fp32` | off | Use fp32 precision instead of fp16 |
| `--preload` | none | Model name to preload at startup |

---

## Endpoints

### `GET /health`

Health check endpoint. Returns service status, device info, and currently loaded models.

**Request:**
```bash
curl http://localhost:8288/health
```

**Response (200 OK):**
```json
{
  "status": "ok",
  "device": "cuda",
  "cuda": {
    "gpu_name": "NVIDIA GeForce RTX 3090",
    "gpu_count": 1,
    "memory_allocated_mb": 256.3,
    "memory_reserved_mb": 512.0
  },
  "models_loaded": [
    ["RealESRGAN_x4plus", 0, false]
  ],
  "fp32_mode": false
}
```

> **Note:** When running on CPU, the `cuda` field will be an empty object `{}` and `device` will be `"cpu"`.

---

### `GET /models`

List all available models, their descriptions, native scale factors, and whether weights have been downloaded.

**Request:**
```bash
curl http://localhost:8288/models
```

**Response (200 OK):**
```json
{
  "models": [
    {
      "name": "RealESRGAN_x4plus",
      "description": "General-purpose x4 upscaler (RRDB, 23 blocks). Best quality for photos.",
      "scale": 4,
      "weights_downloaded": true
    },
    {
      "name": "RealESRNet_x4plus",
      "description": "General-purpose x4 upscaler (RRDB, 23 blocks). PSNR-oriented.",
      "scale": 4,
      "weights_downloaded": false
    },
    {
      "name": "RealESRGAN_x4plus_anime_6B",
      "description": "Optimised for anime images. Smaller model (6 blocks).",
      "scale": 4,
      "weights_downloaded": false
    },
    {
      "name": "RealESRGAN_x2plus",
      "description": "General-purpose x2 upscaler (RRDB, 23 blocks).",
      "scale": 2,
      "weights_downloaded": false
    },
    {
      "name": "realesr-animevideov3",
      "description": "Compact model for anime video frames (VGG-style, 16 convs).",
      "scale": 4,
      "weights_downloaded": false
    },
    {
      "name": "realesr-general-x4v3",
      "description": "Compact general model with denoising control (VGG-style, 32 convs).",
      "scale": 4,
      "weights_downloaded": false
    }
  ]
}
```

---

### `POST /upscale`

Upscale a single image. The image is sent as a multipart form upload and the upscaled image is returned directly as the response body.

**Content-Type:** `multipart/form-data`

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | **Yes** | — | Image file to upscale (JPEG, PNG, WebP, BMP, TIFF) |
| `model_name` | string | No | `RealESRGAN_x4plus` | Model to use (see `/models` for options) |
| `outscale` | float | No | `4` | Final output scale factor. Can differ from native model scale. |
| `face_enhance` | string | No | `false` | Enable GFPGAN face enhancement (`true`/`false`) |
| `tile` | int | No | `0` | Tile size for processing. Use `128` or `256` for large images to avoid OOM. `0` = no tiling. |
| `denoise_strength` | float | No | `0.5` | Denoise strength, 0–1. Only applies to `realesr-general-x4v3`. |
| `output_format` | string | No | `png` | Output format: `png`, `jpg`, `jpeg`, or `webp` |

#### Example — curl

```bash
curl -X POST http://localhost:8288/upscale \
  -F "file=@photo.jpg" \
  -F "model_name=RealESRGAN_x4plus" \
  -F "outscale=4" \
  -F "output_format=png" \
  -o upscaled.png
```

#### Example — Python

```python
import requests

response = requests.post(
    "http://localhost:8288/upscale",
    files={"file": ("photo.jpg", open("photo.jpg", "rb"), "image/jpeg")},
    data={
        "model_name": "RealESRGAN_x4plus",
        "outscale": "4",
        "output_format": "png",
    },
)

if response.status_code == 200:
    with open("upscaled.png", "wb") as f:
        f.write(response.content)
    print("Success!")
else:
    print(f"Error: {response.json()}")
```

#### Example — Anime Image

```bash
curl -X POST http://localhost:8288/upscale \
  -F "file=@anime_art.png" \
  -F "model_name=RealESRGAN_x4plus_anime_6B" \
  -o anime_upscaled.png
```

#### Example — With Face Enhancement

```bash
curl -X POST http://localhost:8288/upscale \
  -F "file=@portrait.jpg" \
  -F "face_enhance=true" \
  -F "outscale=2" \
  -o portrait_enhanced.png
```

#### Example — Large Image (with tiling)

```bash
curl -X POST http://localhost:8288/upscale \
  -F "file=@large_photo.jpg" \
  -F "tile=256" \
  -o large_upscaled.png
```

#### Response — Success (200 OK)

Returns the upscaled image as binary data.

| Header | Value |
|--------|-------|
| `Content-Type` | `image/png`, `image/jpeg`, or `image/webp` |
| `Content-Disposition` | `attachment; filename=upscaled.png` |

#### Response — Error (400 / 500)

```json
{
  "error": "Description of what went wrong."
}
```

Common errors:

| Status | Cause |
|--------|-------|
| 400 | No file provided, empty filename, invalid model name, bad parameter values |
| 400 | Uploaded file cannot be decoded as an image |
| 413 | File exceeds 500 MB upload limit |
| 500 | GPU out of memory (hint: try `tile=256`) |
| 500 | Other inference failures |

---

### `POST /upscale/video`

Upscale a video file frame-by-frame. The video is sent as a multipart form upload and the upscaled MP4 is returned as the response body.

**Content-Type:** `multipart/form-data`

> **Note:** Video upscaling is significantly slower than image upscaling as each frame is processed individually. A 10-second, 30fps video has 300 frames.

#### Parameters

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | **Yes** | — | Video file to upscale (MP4, AVI, MKV, etc.) |
| `model_name` | string | No | `realesr-animevideov3` | Model to use |
| `outscale` | float | No | `4` | Final output scale factor |
| `face_enhance` | string | No | `false` | Enable GFPGAN face enhancement |
| `tile` | int | No | `0` | Tile size for processing |
| `denoise_strength` | float | No | `0.5` | Denoise strength (only for `realesr-general-x4v3`) |

#### Example — curl

```bash
curl -X POST http://localhost:8288/upscale/video \
  -F "file=@input_video.mp4" \
  -F "model_name=realesr-animevideov3" \
  -F "outscale=4" \
  -o upscaled_video.mp4
```

#### Example — Python

```python
import requests

with open("input_video.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8288/upscale/video",
        files={"file": ("input.mp4", f, "video/mp4")},
        data={
            "model_name": "realesr-animevideov3",
            "outscale": "4",
        },
        timeout=3600,  # Videos can take a long time
    )

if response.status_code == 200:
    with open("upscaled_video.mp4", "wb") as f:
        f.write(response.content)
```

#### Response — Success (200 OK)

| Header | Value |
|--------|-------|
| `Content-Type` | `video/mp4` |
| `Content-Disposition` | `attachment; filename=upscaled.mp4` |

#### Response — Error (400 / 500)

Same error format as `/upscale`.

---

## Model Selection Guide

| Model | Best For | Speed | Quality | Native Scale |
|-------|----------|-------|---------|--------------|
| `RealESRGAN_x4plus` | Real-world photos | Slow | ★★★★★ | 4x |
| `RealESRNet_x4plus` | Photos (PSNR-oriented) | Slow | ★★★★☆ | 4x |
| `RealESRGAN_x2plus` | Photos (2x upscale) | Slow | ★★★★★ | 2x |
| `RealESRGAN_x4plus_anime_6B` | Anime/illustration images | Medium | ★★★★★ | 4x |
| `realesr-animevideov3` | Anime video frames | Fast | ★★★★☆ | 4x |
| `realesr-general-x4v3` | General (with denoise control) | Fast | ★★★★☆ | 4x |

### Tips

- **First request** for a model will download weights (~64 MB) and load them into memory. Subsequent requests reuse the cached model.
- Use `--preload` at startup to avoid cold-start latency.
- Use `tile=128` or `tile=256` if you get CUDA out-of-memory errors on large images.
- The `outscale` parameter can be any value (e.g., `3.5`). If it differs from the native model scale, the output is resized with LANCZOS4 interpolation.
- `denoise_strength` only affects the `realesr-general-x4v3` model. `0` = keep noise, `1` = strong denoise.

---

## Error Handling

All error responses follow this JSON format:

```json
{
  "error": "Human-readable error message."
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (missing file, invalid parameters) |
| 404 | Unknown endpoint (response includes list of valid endpoints) |
| 413 | Upload too large (max 500 MB) |
| 500 | Server error (inference failure, OOM, etc.) |

---

## Architecture Notes

- **Model caching:** Models are loaded lazily on first request and cached in memory by `(model_name, tile, half)` key.
- **Thread safety:** A threading lock serialises GPU inference — one request at a time. Concurrent requests will queue.
- **Max upload size:** 500 MB (configurable in `web_service.py`).
- **Temp files:** Video processing uses temporary directories under `tmp_video_jobs/`, cleaned up automatically after each response.
- **Auto-download:** Model weights are downloaded automatically from GitHub releases on first use if not already present in the `weights/` directory.
