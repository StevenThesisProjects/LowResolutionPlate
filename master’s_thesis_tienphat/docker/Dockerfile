# MF-PlateReader, CPU only. The build context is the folder that holds BOTH sibling
# folders restran/ (model code) and masters_thesis_tienphat/ (this app + model/ checkpoint):
#   cd <folder with restran/ and masters_thesis_tienphat/>
#   docker build -f masters_thesis_tienphat/docker/Dockerfile -t mfplate:1.0 .
# or use docker-compose.yml in this folder. Dockerfile.dockerignore (next to this file)
# sends only the files copied below, so the 20 GB of datasets and checkpoints (and
# anything else in that folder) stay out of the build context.

# Stage 1: build the React web UI (served by the API at /)
FROM node:22-slim AS webui
WORKDIR /webui
COPY masters_thesis_tienphat/webui/package.json masters_thesis_tienphat/webui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY masters_thesis_tienphat/webui/ ./
RUN npm run build

# Stage 2: the API (Python, CPU)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MFPLATE_REPO_ROOT=/app \
    MFPLATE_DB=/data/mfplate.sqlite3 \
    MFPLATE_DEVICE=cpu \
    GRADIO_ANALYTICS_ENABLED=False

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
COPY masters_thesis_tienphat/docker/requirements-docker.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /app
# Model code, unchanged from the training repository (same layout as restran/)
COPY restran/src/ /app/src/
COPY restran/configs/ /app/configs/
# The frozen checkpoint, from the app's model/ folder (SHA-256 checked again at start-up
# against serving_config.json)
COPY masters_thesis_tienphat/model/restran_lockbox_combo3_s42_ep30.pth /app/masters_thesis_tienphat/model/
# App
COPY masters_thesis_tienphat/mfplate/ /app/masters_thesis_tienphat/mfplate/
COPY masters_thesis_tienphat/ui/ /app/masters_thesis_tienphat/ui/
COPY --from=webui /webui/dist/ /app/masters_thesis_tienphat/webui/dist/
COPY masters_thesis_tienphat/artifacts/serving_config.json \
     masters_thesis_tienphat/artifacts/decoder_decision.json \
     masters_thesis_tienphat/artifacts/threshold_calibration.json \
     /app/masters_thesis_tienphat/artifacts/

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
USER app
WORKDIR /app/masters_thesis_tienphat
EXPOSE 8000 7860

# Default: the API, which also serves the React UI at /. The optional Gradio UI
# (profile "gradio" in docker-compose.yml) overrides the command.
CMD ["uvicorn", "mfplate.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
