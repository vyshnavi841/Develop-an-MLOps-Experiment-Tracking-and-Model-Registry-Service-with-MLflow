import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Local imports
try:
    from config.settings import settings
    from api.schemas import (
        TrainRequest,
        TrainResponse,
        ExperimentSummary,
        BestRunSummary,
        ModelVersionInfo,
        TransitionStageRequest,
        PredictionRequest,
        PredictionResponse,
        HealthResponse,
    )
    from ml_core.train import train
    from ml_core.run_experiments import run_experiment_sweep, find_best_run_and_register
except ImportError:
    from ..config.settings import settings
    from .schemas import (
        TrainRequest,
        TrainResponse,
        ExperimentSummary,
        BestRunSummary,
        ModelVersionInfo,
        TransitionStageRequest,
        PredictionRequest,
        PredictionResponse,
        HealthResponse,
    )
    from ..ml_core.train import train
    from ..ml_core.run_experiments import run_experiment_sweep, find_best_run_and_register

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("api.main")

# FastAPI App Instance
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-ready MLOps Experiment Tracking & Model Registry Service powered by MLflow and FastAPI.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model Cache: {(model_name, stage_or_alias): (loaded_model, model_version, load_timestamp)}
MODEL_CACHE: Dict[str, Any] = {}


def get_mlflow_client() -> MlflowClient:
    """Initialize and return an MLflow Tracking Client."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def load_model_from_registry(
    model_name: str = "MNISTClassifier",
    stage: str = "Production",
    alias: Optional[str] = None
) -> Tuple[Any, str, str]:
    """
    Load a model from the MLflow Model Registry with memory caching and fallback.
    Returns (loaded_pyfunc_model, actual_version, actual_stage).
    """
    client = get_mlflow_client()
    cache_key = f"{model_name}:{alias or stage}"

    # Check cache freshness (cached for up to 60 seconds)
    if cache_key in MODEL_CACHE:
        cached_model, version, cached_time = MODEL_CACHE[cache_key]
        if time.time() - cached_time < 60:
            return cached_model, version, stage

    # Determine model URI (by alias or stage)
    model_uri = None
    actual_version = "latest"

    try:
        if alias:
            model_uri = f"models:/{model_name}@{alias}"
            mv = client.get_model_version_by_alias(model_name, alias)
            actual_version = str(mv.version)
        else:
            # Try specified stage (e.g. 'Production')
            model_uri = f"models:/{model_name}/{stage}"
            latest_versions = client.get_latest_versions(model_name, stages=[stage])
            if latest_versions:
                actual_version = str(latest_versions[0].version)
            else:
                # Fallback to Staging or any latest version
                fallback_stage = "Staging"
                fallback_versions = client.get_latest_versions(model_name, stages=[fallback_stage])
                if fallback_versions:
                    logger.warning(f"No model in '{stage}', falling back to '{fallback_stage}'")
                    model_uri = f"models:/{model_name}/{fallback_stage}"
                    actual_version = str(fallback_versions[0].version)
                    stage = fallback_stage
                else:
                    # Fallback to latest registered version regardless of stage
                    all_versions = client.get_latest_versions(model_name)
                    if all_versions:
                        actual_version = str(all_versions[0].version)
                        model_uri = f"models:/{model_name}/{actual_version}"
                        stage = all_versions[0].current_stage or "None"
                    else:
                        raise RuntimeError(f"No versions found for registered model '{model_name}'")

        logger.info(f"Loading MLflow model from URI: {model_uri} (Version: {actual_version})")
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        MODEL_CACHE[cache_key] = (loaded_model, actual_version, time.time())
        return loaded_model, actual_version, stage

    except Exception as e:
        logger.error(f"Error loading model '{model_name}' from registry: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not load model '{model_name}' (stage: {stage}, alias: {alias}): {str(e)}"
        )


# Background task helper
def run_training_task(
    experiment_name: str,
    run_name: Optional[str],
    model_type: str,
    hyperparams: Dict[str, Any],
    subset_size: int,
    random_state: int
):
    try:
        logger.info(f"[BackgroundTask] Starting training for experiment '{experiment_name}'...")
        train(
            experiment_name=experiment_name,
            run_name=run_name,
            model_type=model_type,
            hyperparams=hyperparams,
            subset_size=subset_size,
            random_state=random_state
        )
        logger.info(f"[BackgroundTask] Completed training for '{experiment_name}'")
    except Exception as e:
        logger.error(f"[BackgroundTask] Training failed: {e}", exc_info=True)


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Service health and connectivity probe."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.MLFLOW_TRACKING_URI)
    mlflow_ok = False
    model_info = {}

    try:
        client = get_mlflow_client()
        # Test MLflow connectivity
        experiments = client.search_experiments(max_results=1)
        mlflow_ok = True

        # Check registered model status
        try:
            models = client.search_registered_models(max_results=5)
            model_info = {
                "registered_models_count": len(models),
                "default_model": settings.REGISTERED_MODEL_NAME,
                "models": [m.name for m in models]
            }
        except Exception:
            model_info = {"status": "Model registry queryable"}
    except Exception as e:
        logger.warning(f"MLflow health check check warning: {e}")

    return HealthResponse(
        status="healthy" if mlflow_ok else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        mlflow_tracking_uri=tracking_uri,
        mlflow_connected=mlflow_ok,
        postgres_connected=mlflow_ok,
        registered_model_status=model_info
    )


@app.post("/train", response_model=TrainResponse, status_code=status.HTTP_202_ACCEPTED, tags=["Training"])
def trigger_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """
    Accept model configuration and asynchronously trigger a new training run.
    Returns the generated MLflow run ID and experiment ID upon initiation.
    """
    client = get_mlflow_client()

    # Get or create experiment
    experiment = client.get_experiment_by_name(request.experiment_name)
    if not experiment:
        experiment_id = client.create_experiment(request.experiment_name)
        logger.info(f"Created new MLflow experiment: {request.experiment_name} (ID={experiment_id})")
    else:
        experiment_id = experiment.experiment_id

    # Create run placeholder to return immediate Run ID
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=request.run_name or f"{request.model_type}_{int(time.time())}",
        tags={
            "status": "QUEUED",
            "model_type": request.model_type,
            "initiated_by": "FastAPI_POST_train"
        }
    )
    run_id = run.info.run_id

    # Queue training in background
    background_tasks.add_task(
        run_training_task,
        experiment_name=request.experiment_name,
        run_name=request.run_name,
        model_type=request.model_type,
        hyperparams=request.hyperparameters,
        subset_size=request.subset_size,
        random_state=request.random_state
    )

    return TrainResponse(
        message="Model training successfully queued as asynchronous background task.",
        run_id=run_id,
        experiment_id=experiment_id,
        status="ACCEPTED",
        model_type=request.model_type,
        experiment_name=request.experiment_name
    )


@app.get("/experiments", response_model=List[ExperimentSummary], tags=["Experiment Tracking"])
def list_experiments(
    max_results: int = Query(default=50, ge=1, le=1000)
):
    """
    Query MLflow Tracking Server for all experiments, including their IDs, names,
    and a summary of their best runs (ID, metrics, parameters).
    """
    try:
        client = get_mlflow_client()
        experiments = client.search_experiments(max_results=max_results)
        summaries = []

        for exp in experiments:
            # Query runs for this experiment
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                order_by=["metrics.accuracy DESC", "metrics.f1_score DESC"],
                max_results=100
            )

            best_run_summary = None
            if runs:
                best_run = runs[0]
                # Extract primary metric
                acc = best_run.data.metrics.get("accuracy")
                f1 = best_run.data.metrics.get("f1_score")
                metric_name = "accuracy" if acc is not None else ("f1_score" if f1 is not None else "loss")
                metric_value = acc if acc is not None else (f1 if f1 is not None else 0.0)

                best_run_summary = BestRunSummary(
                    run_id=best_run.info.run_id,
                    run_name=best_run.data.tags.get("mlflow.runName", "Unnamed"),
                    metric_name=metric_name,
                    metric_value=metric_value,
                    parameters=best_run.data.params,
                    start_time=best_run.info.start_time
                )

            summaries.append(
                ExperimentSummary(
                    experiment_id=exp.experiment_id,
                    name=exp.name,
                    artifact_location=exp.artifact_location,
                    lifecycle_stage=exp.lifecycle_stage,
                    total_runs=len(runs),
                    best_run=best_run_summary
                )
            )

        return summaries
    except Exception as e:
        logger.error(f"Error querying MLflow experiments: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query MLflow Tracking Server: {str(e)}"
        )


@app.get("/experiments/{experiment_id}/runs", tags=["Experiment Tracking"])
def get_experiment_runs(
    experiment_id: str,
    max_results: int = Query(default=50, ge=1, le=500)
):
    """Retrieve detailed runs for a specific experiment."""
    try:
        client = get_mlflow_client()
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            max_results=max_results,
            order_by=["attribute.start_time DESC"]
        )
        return [
            {
                "run_id": r.info.run_id,
                "run_name": r.data.tags.get("mlflow.runName"),
                "status": r.info.status,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": r.data.tags
            }
            for r in runs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelVersionInfo], tags=["Model Registry"])
def list_registered_models(
    model_name: Optional[str] = Query(default=None, description="Filter by model name")
):
    """
    Query MLflow Model Registry for all registered model versions, including their stages
    (e.g., 'Staging', 'Production', 'Archived'), creation timestamps, and run IDs.
    """
    try:
        client = get_mlflow_client()
        model_versions_out = []

        if model_name:
            versions = client.search_model_versions(f"name='{model_name}'")
        else:
            versions = client.search_model_versions("")

        for mv in versions:
            aliases = []
            try:
                # Fetch aliases if supported by MLflow client
                rm = client.get_registered_model(mv.name)
                for alias_key, ver in rm.aliases.items():
                    if str(ver) == str(mv.version):
                        aliases.append(alias_key)
            except Exception:
                pass

            model_versions_out.append(
                ModelVersionInfo(
                    name=mv.name,
                    version=str(mv.version),
                    current_stage=mv.current_stage or "None",
                    run_id=mv.run_id or "",
                    status=mv.status or "READY",
                    creation_timestamp=mv.creation_timestamp,
                    last_updated_timestamp=mv.last_updated_timestamp,
                    description=mv.description,
                    aliases=aliases,
                    tags=mv.tags or {}
                )
            )

        return model_versions_out
    except Exception as e:
        logger.error(f"Error querying Model Registry: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query MLflow Model Registry: {str(e)}"
        )


@app.post("/models/transition", response_model=Dict[str, Any], tags=["Model Registry"])
def transition_model_stage(request: TransitionStageRequest):
    """
    Transition a model version's lifecycle stage (e.g. Staging -> Production).
    """
    try:
        client = get_mlflow_client()
        updated_version = client.transition_model_version_stage(
            name=request.model_name,
            version=request.version,
            stage=request.stage,
            archive_existing_versions=request.archive_existing_versions
        )

        # Clear cache so new stage model is loaded immediately
        MODEL_CACHE.clear()

        # Update alias if transitioning to Production
        if request.stage.lower() == "production":
            try:
                client.set_registered_model_alias(request.model_name, "champion", request.version)
            except Exception:
                pass

        return {
            "status": "SUCCESS",
            "message": f"Successfully transitioned '{request.model_name}' version {request.version} to stage '{request.stage}'",
            "model_name": request.model_name,
            "version": str(updated_version.version),
            "stage": updated_version.current_stage
        }
    except Exception as e:
        logger.error(f"Error transitioning model stage: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to transition model stage: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Perform real-time digit classification inference.
    Loads the 'Production' staged version of 'MNISTClassifier' from MLflow Model Registry
    and scores the input 784 flattened pixel array.
    """
    # 1. Parse & Validate input data
    if request.image_data is not None:
        input_array = np.array(request.image_data, dtype=np.float32).reshape(1, 784)
    elif request.images is not None:
        input_array = np.array(request.images, dtype=np.float32)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either 'image_data' (single 784-length array) or 'images' (batch) must be provided."
        )

    # Normalize pixels to [0.0, 1.0] if range is [0, 255]
    if input_array.max() > 1.0:
        input_array = input_array / 255.0

    # 2. Load Model from MLflow Registry
    model, version, actual_stage = load_model_from_registry(
        model_name=request.model_name,
        stage=request.stage,
        alias=request.alias
    )

    # 3. Perform Inference & Measure Latency
    start_time = time.perf_counter()
    try:
        raw_preds = model.predict(input_array)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Handle various model output formats (numpy arrays, pandas df, probabilities, class labels)
        if isinstance(raw_preds, np.ndarray):
            if raw_preds.ndim == 2 and raw_preds.shape[1] > 1:
                # Softmax probabilities output
                pred_classes = np.argmax(raw_preds, axis=1).tolist()
                probs = raw_preds.tolist()
            else:
                pred_classes = [int(p) for p in raw_preds.flatten()]
                probs = None
        else:
            # Fallback for scalar/list outputs
            pred_classes = [int(p) for p in np.array(raw_preds).flatten()]
            probs = None

        return PredictionResponse(
            predictions=pred_classes,
            probabilities=probs,
            model_name=request.model_name,
            model_version=version,
            model_stage=actual_stage,
            inference_time_ms=round(duration_ms, 2),
            input_count=len(input_array)
        )

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed on model '{request.model_name}': {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=True)
