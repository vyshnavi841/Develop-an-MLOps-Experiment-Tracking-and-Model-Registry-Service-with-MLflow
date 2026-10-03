# MLOps Experiment Tracking & Model Registry Service with MLflow

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.11.3-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-13-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg?logo=pytest&logoColor=white)](https://docs.pytest.org)

A production-grade, containerized MLOps service for managing the complete lifecycle of Machine Learning experiments—from tracking runs and logging artifacts to registering, versioning, and deploying models with **MLflow**, **PostgreSQL**, and **FastAPI**.

---

## 📋 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Prerequisites & Environment](#-prerequisites--environment)
- [Quick Start with Docker Compose](#-quick-start-with-docker-compose)
- [Running ML Experiments & Model Registry](#-running-ml-experiments--model-registry)
- [REST API Reference & Endpoints](#-rest-api-reference--endpoints)
  - [1. Service Health Check](#1-health-check)
  - [2. Trigger Training Run](#2-trigger-training-run)
  - [3. List Experiments & Summaries](#3-list-experiments--summaries)
  - [4. Query Registered Models](#4-query-registered-models)
  - [5. Transition Model Stage](#5-transition-model-stage)
  - [6. Perform Real-Time Inference](#6-perform-real-time-inference)
- [Running Tests](#-running-tests)
- [Design Decisions & Best Practices](#-design-decisions--best-practices)

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Client Layer
        CLIENT[Client / Application / UI]
        CLI[MLOps CLI / Pipeline Runner]
        MLFLOW_UI[MLflow Native Web UI :5000]
    end

    subgraph Containerized Services [docker-compose]
        subgraph FastAPI Gateway [:8000]
            API_TRAIN["POST /train (BackgroundTasks)"]
            API_EXP["GET /experiments"]
            API_MOD["GET /models"]
            API_TRANS["POST /models/transition"]
            API_PRED["POST /predict"]
            CACHE[(In-Memory Model Cache)]
        end

        subgraph MLflow Server [:5000]
            TRACKING[MLflow Tracking Service]
            REGISTRY[MLflow Model Registry]
        end

        subgraph Storage Backends
            POSTGRES[(PostgreSQL Database :5432)]
            ARTIFACTS[(Local / Shared MLruns Volume)]
        end
    end

    subgraph ML Core
        TRAIN_PY[ml_core/train.py]
        SWEEP_PY[ml_core/run_experiments.py]
    end

    CLI --> SWEEP_PY
    SWEEP_PY --> TRAIN_PY
    TRAIN_PY -->|Log Params, Metrics, Artifacts| TRACKING
    TRAIN_PY -->|Save Models & Plots| ARTIFACTS
    SWEEP_PY -->|Register Best Model Version| REGISTRY

    CLIENT --> FastAPI Gateway
    API_TRAIN -->|Async Spawn| TRAIN_PY
    API_EXP & API_MOD & API_TRANS <--> MLflow Server
    API_PRED --> CACHE
    CACHE -.->|Cache Miss / TTL Refresh| REGISTRY

    TRACKING <--> POSTGRES
    REGISTRY <--> POSTGRES
    TRACKING <--> ARTIFACTS
    MLFLOW_UI <--> MLflow Server
```

---

## 🚀 Key Features

1. **Enterprise MLflow Tracking & Registry Stack**:
   - **PostgreSQL Backend**: High-performance relational metadata storage for runs, metrics, parameters, tags, and stages.
   - **Artifact Store**: Centralized artifact repository for models, signatures, confusion matrices, and metrics reports.
2. **Multi-Model Training Pipeline (`ml_core/train.py`)**:
   - Trains classification models on the **MNIST** dataset (Random Forest, Logistic Regression, Gradient Boosting, CNN / MLP).
   - Full reproducibility with configurable random seeds.
   - Automated evaluation, signature inference, and plot artifact generation (confusion matrix heatmaps, classification reports).
3. **Automated Sweep & Governance (`ml_core/run_experiments.py`)**:
   - Orchestrates $\ge 10$ distinct hyperparameter runs.
   - Programmatically discovers the highest-performing model run via the MLflow Search API.
   - Registers the model into the MLflow Model Registry (`MNISTClassifier`) and transitions it to the `Staging` lifecycle stage.
4. **FastAPI Model Serving & Governance Gateway (`api/main.py`)**:
   - `POST /train`: Asynchronously triggers training via `BackgroundTasks` without blocking.
   - `GET /experiments`: Retrieves experiment summaries and best run metrics.
   - `GET /models`: Lists registered models, versions, and stages.
   - `POST /models/transition`: Promotes models between lifecycle stages (`Staging` $\rightarrow$ `Production`).
   - `POST /predict`: Serves real-time inference using the `Production` registered model with in-memory caching and graceful fallback.
5. **Optimized Multi-Stage Containerization**:
   - Multi-stage Docker builds for minimal container size and fast startup.
   - Production Docker Compose orchestration with dependency healthchecks.

---

## 📂 Project Structure

```text
.
├── docker-compose.yml              # Complete service orchestration (postgres, mlflow, api)
├── Dockerfile.mlflow               # MLflow Tracking Server image definition
├── .env.example                    # Environment variable configuration template
├── requirements.txt                # Python dependencies
├── README.md                       # Comprehensive documentation and guide
├── ARCHITECTURE.md                 # Detailed architecture & governance specification
│
├── config/
│   ├── __init__.py
│   └── settings.py                 # Centralized Pydantic configuration settings
│
├── ml_core/
│   ├── __init__.py
│   ├── train.py                    # MNIST training script with MLflow autologging & artifacts
│   └── run_experiments.py          # Sweep orchestrator & champion model registration
│
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI service with all required REST endpoints
│   ├── schemas.py                  # Pydantic request/response validation schemas
│   └── Dockerfile                  # Multi-stage production Dockerfile for FastAPI
│
└── tests/
    ├── conftest.py                 # Pytest fixtures and test environment setup
    ├── test_api.py                 # Unit tests for FastAPI endpoints and validation
    ├── test_train.py               # Unit tests for training logic and data processing
    └── test_mlflow_integration.py  # End-to-end integration tests for MLflow and Registry
```

---

## ⚙️ Prerequisites & Environment

- **Docker** $\ge 20.10$ & **Docker Compose** $\ge 2.0$
- **Python** $\ge 3.10$ (for local execution / testing)

### Setup Environment Variables
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

---

## 🐳 Quick Start with Docker Compose

Launch the entire stack (PostgreSQL, MLflow Tracking Server, and FastAPI Gateway) with a single command:

```bash
docker-compose up -d --build
```

### Verify Service Status
```bash
docker-compose ps
```

### Access Web Interfaces
- **MLflow Tracking & Model Registry UI**: [http://localhost:5000](http://localhost:5000)
- **FastAPI Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🔬 Running ML Experiments & Model Registry

You can run experiments either via the CLI or directly through the FastAPI endpoint.

### 1. Single Model Training
Execute `train.py` with custom hyperparameters:
```bash
python -m ml_core.train \
  --experiment_name "MNIST-Classification" \
  --model_type "random_forest" \
  --n_estimators 100 \
  --max_depth 15 \
  --subset_size 5000
```

### 2. Hyperparameter Sweep & Automatic Model Registration
Execute `run_experiments.py` to run 10 distinct configurations, select the champion model based on accuracy, and register it to `MNISTClassifier` in `Staging`:
```bash
python -m ml_core.run_experiments \
  --experiment_name "MNIST-Classification" \
  --model_name "MNISTClassifier" \
  --num_runs 10 \
  --metric "accuracy" \
  --stage "Staging"
```

---

## 🌐 REST API Reference & Endpoints

### 1. Health Check
**Endpoint**: `GET /health`

```bash
curl -X GET "http://localhost:8000/health"
```
**Sample Response:**
```json
{
  "status": "healthy",
  "app_name": "MLOps MNIST Experiment Tracking & Model Registry Service",
  "version": "1.0.0",
  "mlflow_tracking_uri": "http://mlflow_server:5000",
  "mlflow_connected": true,
  "postgres_connected": true,
  "registered_model_status": {
    "registered_models_count": 1,
    "default_model": "MNISTClassifier",
    "models": ["MNISTClassifier"]
  }
}
```

---

### 2. Trigger Training Run
**Endpoint**: `POST /train` (Asynchronous background task)

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_name": "MNIST-Classification",
    "model_type": "random_forest",
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 15,
      "min_samples_split": 2
    },
    "subset_size": 5000,
    "random_state": 42
  }'
```
**Sample Response (HTTP 202 Accepted):**
```json
{
  "message": "Model training successfully queued as asynchronous background task.",
  "run_id": "9a38f7129b0148b3b3a4cb10e42f9b8c",
  "experiment_id": "1",
  "status": "ACCEPTED",
  "model_type": "random_forest",
  "experiment_name": "MNIST-Classification"
}
```

---

### 3. List Experiments & Summaries
**Endpoint**: `GET /experiments`

```bash
curl -X GET "http://localhost:8000/experiments"
```
**Sample Response:**
```json
[
  {
    "experiment_id": "1",
    "name": "MNIST-Classification",
    "artifact_location": "/mlruns/1",
    "lifecycle_stage": "active",
    "total_runs": 10,
    "best_run": {
      "run_id": "9a38f7129b0148b3b3a4cb10e42f9b8c",
      "run_name": "rf_trees120_depth18",
      "metric_name": "accuracy",
      "metric_value": 0.968,
      "parameters": {
        "model_type": "random_forest",
        "n_estimators": "120",
        "max_depth": "18"
      },
      "start_time": 1727945000000
    }
  }
]
```

---

### 4. Query Registered Models
**Endpoint**: `GET /models`

```bash
curl -X GET "http://localhost:8000/models"
```
**Sample Response:**
```json
[
  {
    "name": "MNISTClassifier",
    "version": "1",
    "current_stage": "Staging",
    "run_id": "9a38f7129b0148b3b3a4cb10e42f9b8c",
    "status": "READY",
    "creation_timestamp": 1727945100000,
    "last_updated_timestamp": 1727945105000,
    "description": "Model trained with random_forest, accuracy=0.9680",
    "aliases": ["challenger"],
    "tags": {}
  }
]
```

---

### 5. Transition Model Stage
**Endpoint**: `POST /models/transition`

Promote version 1 from `Staging` to `Production`:
```bash
curl -X POST "http://localhost:8000/models/transition" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "MNISTClassifier",
    "version": "1",
    "stage": "Production",
    "archive_existing_versions": true
  }'
```
**Sample Response:**
```json
{
  "status": "SUCCESS",
  "message": "Successfully transitioned 'MNISTClassifier' version 1 to stage 'Production'",
  "model_name": "MNISTClassifier",
  "version": "1",
  "stage": "Production"
}
```

---

### 6. Perform Real-Time Inference
**Endpoint**: `POST /predict`

Send a flattened 784-length array of pixel values:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "image_data": [0, 0, 0, 0, 0, 0.5, 0.9, 0.2, 0, 0, ... (784 floats) ...],
    "model_name": "MNISTClassifier",
    "stage": "Production"
  }'
```
**Sample Response:**
```json
{
  "predictions": [7],
  "probabilities": null,
  "model_name": "MNISTClassifier",
  "model_version": "1",
  "model_stage": "Production",
  "inference_time_ms": 3.45,
  "input_count": 1
}
```

---

## 🧪 Running Tests

### Local Pytest Execution
Run unit and integration tests locally:
```bash
pytest tests/ -v
```

### Running Tests Inside Docker
Execute test suite within the running FastAPI container:
```bash
docker-compose exec api pytest tests/ -v
```

---

## 🛡️ Design Decisions & Best Practices

1. **Decoupled Architecture**:
   - Clear separation between training orchestrators (`ml_core/`) and HTTP API gateway (`api/`).
2. **Model Caching & Zero-Downtime Reload**:
   - `MODEL_CACHE` dynamically caches the model in RAM with a 60-second TTL.
   - Calling `/models/transition` automatically purges the cache to immediately load the new version on next request.
3. **Graceful Fallback Mechanism**:
   - If `/predict` targets `Production` before promotion, the loader automatically falls back to `Staging` or the latest active version.
4. **Reproducibility**:
   - Full deterministic seeding across Python hash, NumPy, and Scikit-Learn.
