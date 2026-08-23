# Cats vs Dogs — End-to-End MLOps Pipeline

Binary image classification pipeline (cats vs dogs) for a pet adoption
platform, covering data versioning, model training, experiment
tracking, containerization, CI, CD, and monitoring.

## Architecture

```
Kaggle Dataset --> DVC-tracked data/ --> Preprocessing (224x224, augment)
    --> CNN Training + MLflow Tracking --> model.pt artifact
    --> Docker Image --> GHCR --> CI: lint+test+build+push
    --> CD: deploy to cluster + smoke test (/health, /predict)
    --> Prometheus + Grafana monitoring
```

## Project Structure
```
catsdogs-mlops/
├── .github/workflows/       # CI (lint/test/build/push) + CD (deploy/smoke test)
├── data/                    # raw/processed images (DVC-tracked, not committed to git)
├── deployment/               # Kubernetes Deployment/Service + ServiceMonitor
├── src/
│   ├── app.py                # FastAPI service: /health, /predict, /metrics
│   ├── config.py              # shared constants (image size, paths, splits)
│   ├── download_data.py       # Kaggle download + 80/10/10 split
│   ├── model.py                # CNN architecture
│   ├── pipelines.py            # image preprocessing/augmentation
│   └── train.py                 # training loop + MLflow logging
├── tests/                    # pytest: preprocessing, model, API
├── dvc.yaml                  # DVC pipeline: download_data -> train
├── Dockerfile                # multi-stage build
└── requirements.txt
```

## 1. Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Data versioning with DVC
This project tracks the dataset with DVC rather than committing images
to git directly.

```bash
dvc init
dvc remote add -d storage <your-remote-url>   # e.g. S3, GDrive, local path
dvc repro                # runs the dvc.yaml pipeline: download_data -> train
dvc push                 # push data + model artifacts to the DVC remote
```

Kaggle API credentials are required for `download_data.py`
(`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars).

## 3. Train directly (without DVC)
```bash
cd src
python download_data.py
python train.py
mlflow ui --backend-store-uri ./mlruns --port 5000   # inspect runs at localhost:5000
```

## 4. Run tests
```bash
pytest tests/ -v
```
Covers: the preprocessing function (`preprocess_image`), the model's
forward/inference path, and the API's `/health` and `/predict`
endpoints (skips real-prediction assertions gracefully if no trained
checkpoint is present yet).

## 5. Docker
```bash
docker build -t catsdogs-api:latest .
docker run -d -p 8000:8000 catsdogs-api:latest

curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -F "file=@/path/to/image.jpg"
```
Swagger UI: http://localhost:8000/docs

## 6. CI/CD (GitHub Actions)
`.github/workflows/mlops-pipeline.yml` runs on every push/PR to `main`:

**CI (`build-test-stage`)**: checkout → install deps → flake8 lint →
pytest → train (using `KAGGLE_USERNAME`/`KAGGLE_KEY` repo secrets if
configured) → upload MLflow/model artifacts → build Docker image →
push to `ghcr.io/<owner>/catsdogs-api`.

**CD (`deploy-and-smoke-test`)**: spins up an ephemeral `kind`
cluster → loads the freshly built image → applies
`deployment/deployment.yaml` → waits for rollout → **smoke-tests
`/health` and `/predict` over the exposed service, failing the
pipeline if either check fails.**

To enable real training in CI, add `KAGGLE_USERNAME` and `KAGGLE_KEY`
as repository secrets (Settings → Secrets and variables → Actions).

## 7. Kubernetes deployment (manual / local cluster)
```bash
minikube start
minikube docker-env | Invoke-Expression   # or: eval $(minikube docker-env) on Linux/macOS
docker build -t catsdogs-api:latest .
kubectl apply -f deployment/deployment.yaml
minikube service catsdogs-service
```
The Deployment defines `readinessProbe`/`livenessProbe` against
`/health`, so Kubernetes will not route traffic to a pod whose model
failed to load, and will restart it automatically.

## 8. Monitoring
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack
kubectl apply -f deployment/servicemonitor.yaml
kubectl port-forward svc/monitoring-grafana 3000:80
```
The API exposes request count, latency, and error metrics at
`/metrics` via `prometheus-fastapi-instrumentator`, scraped every 15s
by the `ServiceMonitor`.

## Notes on reuse
The CI/CD scaffolding (Docker multi-stage pattern, GitHub Actions
structure, Kubernetes manifests, Prometheus/Grafana wiring) is adapted
from an earlier tabular-data MLOps project. The data pipeline, model
architecture, training loop, and inference API are new for this image
classification use case.
