from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator


class TrainRequest(BaseModel):
    experiment_name: str = Field(default="MNIST-Classification", description="Target MLflow experiment name")
    run_name: Optional[str] = Field(default=None, description="Optional custom name for this run")
    model_type: str = Field(default="random_forest", description="Model architecture: 'random_forest', 'logistic_regression', 'gradient_boosting', 'cnn', 'mlp'")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=lambda: {"n_estimators": 100, "max_depth": 15, "min_samples_split": 2},
        description="Dictionary of model hyperparameters"
    )
    subset_size: int = Field(default=5000, ge=100, le=70000, description="Number of MNIST samples to train on")
    random_state: int = Field(default=42, description="Random seed for reproducibility")

    @validator("model_type")
    def validate_model_type(cls, v):
        valid = ["random_forest", "logistic_regression", "gradient_boosting", "cnn", "mlp"]
        if v.lower() not in valid:
            raise ValueError(f"model_type must be one of {valid}")
        return v.lower()


class TrainResponse(BaseModel):
    message: str
    run_id: str
    experiment_id: str
    status: str
    model_type: str
    experiment_name: str


class BestRunSummary(BaseModel):
    run_id: Optional[str] = None
    run_name: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    start_time: Optional[int] = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str
    total_runs: int
    best_run: Optional[BestRunSummary] = None


class ModelVersionInfo(BaseModel):
    name: str
    version: str
    current_stage: str
    run_id: str
    status: str
    creation_timestamp: int
    last_updated_timestamp: Optional[int] = None
    description: Optional[str] = None
    aliases: List[str] = []
    tags: Dict[str, str] = {}


class TransitionStageRequest(BaseModel):
    model_name: str = Field(default="MNISTClassifier", description="Registered model name")
    version: str = Field(..., description="Model version number to transition (e.g. '1')")
    stage: str = Field(..., description="Target stage: 'Staging', 'Production', 'Archived', 'None'")
    archive_existing_versions: bool = Field(default=False, description="Archive other versions in target stage")

    @validator("stage")
    def validate_stage(cls, v):
        valid_stages = ["Staging", "Production", "Archived", "None"]
        for s in valid_stages:
            if v.lower() == s.lower():
                return s
        raise ValueError(f"Stage must be one of {valid_stages}")


class PredictionRequest(BaseModel):
    # Support both single image (784 flattened pixels) and batch images
    image_data: Optional[List[float]] = Field(
        default=None,
        description="Flattened list of 784 pixel values (0-255 or 0.0-1.0) for a single 28x28 digit"
    )
    images: Optional[List[List[float]]] = Field(
        default=None,
        description="Batch of flattened 784-length pixel lists"
    )
    model_name: str = Field(default="MNISTClassifier", description="Name of the model in registry")
    stage: str = Field(default="Production", description="Stage of the model to load ('Production', 'Staging')")
    alias: Optional[str] = Field(default=None, description="Optional model alias to load (e.g. 'champion')")

    @validator("image_data")
    def validate_single_image(cls, v):
        if v is not None and len(v) != 784:
            raise ValueError(f"image_data must contain exactly 784 pixel values, got {len(v)}")
        return v

    @validator("images")
    def validate_batch_images(cls, v):
        if v is not None:
            if len(v) == 0:
                raise ValueError("images batch cannot be empty")
            for idx, img in enumerate(v):
                if len(img) != 784:
                    raise ValueError(f"Image at index {idx} must contain exactly 784 pixels, got {len(img)}")
        return v


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    model_name: str
    model_version: str
    model_stage: str
    inference_time_ms: float
    input_count: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    mlflow_tracking_uri: str
    mlflow_connected: bool
    postgres_connected: bool
    registered_model_status: Dict[str, Any]
