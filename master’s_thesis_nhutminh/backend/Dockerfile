# PlateReader LR API — CPU-only image.
# Build context = master’s_thesis_nhutminh/ (cần src/, configs/, tools/ để load model):
#   docker compose build api
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" "torchvision>=0.17"
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Code model (chỉ đọc) + backend. Checkpoint KHÔNG copy vào image: mount qua volume.
COPY configs/ configs/
COPY src/ src/
COPY tools/ tools/
COPY backend/__init__.py backend/__init__.py
COPY backend/app/ backend/app/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

# 1 worker: model (~120 MB) + hàng đợi batch nằm trong process; scale bằng nhiều container.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
