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

## 2. Data & Code Versioning with Git and Git LFS

Git is used for versioning the source code, project structure, scripts,
notebooks, and configuration files.

Git LFS (Large File Storage) is used to version the dataset and
pre-processed image data without storing large binary files directly
in the normal Git history.

### Git LFS setup

Install and initialize Git LFS:

```bash
git lfs install
```

### Track large dataset files:
```bash
git lfs track "data/raw/**/*.jpg"
git lfs track "data/raw/**/*.png"
git lfs track "data/processed/**/*.jpg"
git lfs track "data/processed/**/*.png"
```
The LFS configuration is stored in .gitattributes.

Add and commit the files:
```bash
git add .gitattributes
git add data/
git commit -m "Add dataset and pre-processed data using Git LFS"
git push
```
To verify files tracked by Git LFS:
```bash
git lfs ls-files
```
To retrieve the LFS files after cloning the repository:
```
git lfs pull
```

Kaggle API credentials are required for `download_data.py`
(`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars).

## 3. Train directly
```bash
cd src
python download_data.py
python train.py

# $env:MLFLOW_ALLOW_FILE_STORE="true"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # inspect runs at localhost:5000
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
curl -X POST http://localhost:8000/predict -F "file=@D:\Github\catsdogs-mlops\data\raw\test\cats\cat.10.jpg"
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
