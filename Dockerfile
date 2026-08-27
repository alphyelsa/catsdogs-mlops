# Stage 1: Build & Python Dependency Wheel Cache
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CPU-only torch/torchvision keep the image reasonable in size; swap
# this index for a CUDA build if deploying to a GPU node pool.
RUN pip install --user --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Stage 2: Clean Executable Runtime
FROM python:3.12-slim AS runner
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /root/.local
COPY ./src ./src
COPY ./model ./model

ENV PATH=/root/.local/bin:$PATH
ENV MODEL_PATH=/app/model/model.pt
ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

WORKDIR /app/src
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
