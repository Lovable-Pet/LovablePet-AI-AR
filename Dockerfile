# Pipeline HTTP server (api/server.py). Templates, job files and model caches
# are not in the image: mount output/ and /cache (deploy/docker-compose.yml).
FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    U2NET_HOME=/cache/u2net

# libgl1/libglib2.0-0 for opencv (rembg), curl for the healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU torch first, in its own layer: it is the big one and changes rarely
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY api/ api/
COPY tests/scripts.py tests/scripts.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS localhost:8000/health || exit 1

# One worker only: jobs live in this process's memory (api/server.py _jobs)
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
