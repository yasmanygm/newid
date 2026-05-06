# Passport MRZ Reader — HTTP Service

FastAPI service wrapping the MRZ reader pipeline (`passport_reader_no_train`). Accepts passport/ID images, returns parsed MRZ fields plus ICAO 9303 validity. Runs entirely on CPU, fully offline.

## Quick start

```bash
docker compose up --build -d

# Wait for readiness (first call may take ~60-90s while models load + warm up)
curl --retry 20 --retry-delay 3 --retry-connrefused -fsS http://localhost:8000/ready

# Upload an image
curl -s -F file=@test.tif http://localhost:8000/v1/mrz | jq .

# Or send base64
base64 -w0 test.tif | jq -Rs '{image:.}' \
  | curl -s -H 'content-type: application/json' -d @- \
         http://localhost:8000/v1/mrz/base64 | jq .

# OpenAPI docs
open http://localhost:8000/docs
```

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/mrz` | multipart upload (`file=@...`) |
| `POST` | `/v1/mrz/base64` | JSON `{"image": "<base64>"}` |
| `GET`  | `/health` | liveness (always 200 once the process is up) |
| `GET`  | `/ready` | readiness — 503 until models loaded + warmed up |
| `GET`  | `/version` | build info |
| `GET`  | `/docs`, `/openapi.json` | OpenAPI |

### Response schema

```json
{
  "document_type": "TD3",
  "valid": true,
  "mrz": ["P<URYSMITH<<JOHN<<...", "ABC123456<7URY800..."],
  "fields": {
    "surname": "SMITH",
    "name": "JOHN",
    "country": "URY",
    "nationality": "URY",
    "birth_date": "800101",
    "expiry_date": "300101",
    "sex": "M",
    "document_number": "ABC123456"
  },
  "processing_time_s": 7.12,
  "request_id": "c0a8..."
}
```

When the pipeline finds no MRZ the response is HTTP 200 with `valid: false`, `mrz: null`, `error: "no_mrz_found"`. Server errors (inference crashes) return HTTP 500 with an `inference_failed` envelope.

## Configuration

Environment variables (set on `docker compose`, `docker run -e`, or a `.env` file):

| Variable | Default | Meaning |
|---|---|---|
| `MRZ_ENGINE` | `paddleocr` | `paddleocr` or `fastmrz`. `fastmrz` requires the image built with `ENABLE_FASTMRZ=1` and `models/tessdata/` populated. |
| `MAX_UPLOAD_MB` | `10` | Rejects payloads above this size with HTTP 413. |
| `LOG_LEVEL` | `INFO` | Root logger level (`DEBUG`, `INFO`, `WARNING`). |
| `GIT_SHA` | `dev` | Shown on `/version`. Set at build time via build-arg. |

## Build variants

### Default (paddleocr only, smaller image)

```bash
docker compose build
```

### With FastMRZ engine (adds tesseract)

```bash
# 1. Populate tessdata into the build context first
cp -r /path/to/mrz.traineddata models/tessdata/
# 2. Build with the flag
docker compose build --build-arg ENABLE_FASTMRZ=1
# 3. Run with the fastmrz engine
MRZ_ENGINE=fastmrz docker compose up -d
```

## Architecture and decisions

- **Single uvicorn worker + `asyncio.Lock`**: PaddleOCR's `Predictor` keeps state on the instance, so concurrent inference isn't safe. We serialize `pipeline.process` under a lock and run it in a thread via `asyncio.to_thread`. The event loop stays responsive for `/health` and `/ready` while a request is in flight. To scale, run more containers — not more workers.
- **Models baked into the image**: ~21MB total (YOLO 5.5MB, PaddleOCR det 4.8MB, rec 11MB). Self-contained, air-gap-deployable.
- **Warmup at startup**: the lifespan handler runs `pipeline.process` on a synthetic blank image so PaddleOCR finishes its first-call JIT compile before `/ready` returns 200. Startup takes ~60-90s end to end.
- **Hardened runtime**: non-root user (`app`, uid 10001), `read_only: true` rootfs, tmpfs for `/tmp` and `/app/.ultralytics`, `cap_drop: ALL`, `no-new-privileges`, 4GB memory cap.
- **Offline by construction**: `YOLO_OFFLINE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` set at runtime. Nothing should reach out to github/pypi during request handling. You can prove this by starting with `--network none` after the first build.

## Local development (without Docker)

```bash
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

## Verification checklist

```bash
# Liveness
curl -fsS localhost:8000/health

# Readiness (503 initially, 200 after warmup)
curl -i localhost:8000/ready

# Version
curl -s localhost:8000/version | jq .

# Single inference
curl -s -F file=@test.tif localhost:8000/v1/mrz | jq .

# Oversize (should return 413)
head -c 20M /dev/urandom > big.bin
curl -i -F file=@big.bin localhost:8000/v1/mrz

# Bad base64
curl -i -H 'content-type: application/json' -d '{"image":"not-base64"}' \
     localhost:8000/v1/mrz/base64

# Concurrency: lock must serialize — total wall time ≈ 4 × single-call time
time (for i in $(seq 1 4); do
        curl -s -F file=@test.tif localhost:8000/v1/mrz >/dev/null &
      done; wait)
```

## Non-goals (and how to address them)

- **Authentication**: add a reverse proxy (nginx / traefik / cloud API gateway) or a small bearer-token middleware.
- **Batch endpoint**: unhelpful given the serial lock; fan out with an async queue instead.
- **Multi-worker / GPU**: start multiple containers and put a load balancer in front; for GPU, switch the base image and install `paddlepaddle-gpu`.
