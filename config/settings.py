import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "MLOps MNIST Experiment Tracking & Model Registry Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PORT: int = Field(default=8000, env="API_PORT")

    # PostgreSQL Database
    POSTGRES_USER: str = Field(default="mlflow_user", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(default="mlflow_password", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field(default="mlflow_db", env="POSTGRES_DB")
    POSTGRES_HOST: str = Field(default="postgres", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    DATABASE_URL: str = Field(
        default="postgresql://mlflow_user:mlflow_password@postgres:5432/mlflow_db",
        env="DATABASE_URL"
    )

    # MLflow Tracking & Registry
    MLFLOW_TRACKING_URI: str = Field(default="http://mlflow_server:5000", env="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field(default="MNIST-Classification", env="MLFLOW_EXPERIMENT_NAME")
    REGISTERED_MODEL_NAME: str = Field(default="MNISTClassifier", env="REGISTERED_MODEL_NAME")
    DEFAULT_MODEL_STAGE: str = Field(default="Production", env="DEFAULT_MODEL_STAGE")
    FALLBACK_MODEL_STAGE: str = Field(default="Staging", env="FALLBACK_MODEL_STAGE")

    # Artifact Storage
    ARTIFACTS_DESTINATION: str = Field(default="/mlruns", env="ARTIFACTS_DESTINATION")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
