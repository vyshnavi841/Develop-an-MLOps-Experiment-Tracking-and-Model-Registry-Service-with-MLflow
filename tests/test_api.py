from unittest.mock import patch, MagicMock
import pytest
from fastapi import status


def test_health_endpoint(test_client):
    """Test the /health endpoint probe."""
    response = test_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert "app_name" in data
    assert "version" in data
    assert "mlflow_tracking_uri" in data


def test_predict_validation_invalid_length(test_client):
    """Test /predict validation rejects image vectors that are not length 784."""
    payload = {
        "image_data": [0.5] * 100,  # Invalid length (100 instead of 784)
        "model_name": "MNISTClassifier",
        "stage": "Production"
    }
    response = test_client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_validation_empty_body(test_client):
    """Test /predict validation rejects empty request with neither image_data nor images."""
    payload = {
        "model_name": "MNISTClassifier",
        "stage": "Production"
    }
    response = test_client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_train_validation_invalid_model_type(test_client):
    """Test /train endpoint rejects unsupported model architectures."""
    payload = {
        "experiment_name": "test_exp",
        "model_type": "quantum_neural_net",  # Unsupported
        "subset_size": 1000
    }
    response = test_client.post("/train", json=payload)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_train_endpoint_success(test_client):
    """Test /train endpoint queues background task and returns run_id."""
    with patch("api.main.get_mlflow_client") as mock_client_getter:
        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp-123"
        mock_client.get_experiment_by_name.return_value = mock_exp
        
        mock_run = MagicMock()
        mock_run.info.run_id = "run-abc-789"
        mock_client.create_run.return_value = mock_run

        mock_client_getter.return_value = mock_client

        payload = {
            "experiment_name": "MNIST-Test-Experiment",
            "model_type": "random_forest",
            "hyperparameters": {"n_estimators": 50, "max_depth": 10},
            "subset_size": 500
        }
        response = test_client.post("/train", json=payload)
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert data["run_id"] == "run-abc-789"
        assert data["experiment_id"] == "exp-123"
        assert data["status"] == "ACCEPTED"


def test_experiments_endpoint_mocked(test_client):
    """Test GET /experiments endpoint returns structured summaries."""
    with patch("api.main.get_mlflow_client") as mock_client_getter:
        mock_client = MagicMock()
        
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_exp.name = "MNIST-Classification"
        mock_exp.artifact_location = "/mlruns/1"
        mock_exp.lifecycle_stage = "active"

        mock_run = MagicMock()
        mock_run.info.run_id = "run-001"
        mock_run.info.start_time = 1600000000
        mock_run.data.metrics = {"accuracy": 0.965, "f1_score": 0.964}
        mock_run.data.params = {"n_estimators": "100"}
        mock_run.data.tags = {"mlflow.runName": "rf_champion"}

        mock_client.search_experiments.return_value = [mock_exp]
        mock_client.search_runs.return_value = [mock_run]
        mock_client_getter.return_value = mock_client

        response = test_client.get("/experiments")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "MNIST-Classification"
        assert data[0]["best_run"]["metric_value"] == 0.965
        assert data[0]["best_run"]["run_id"] == "run-001"


def test_models_endpoint_mocked(test_client):
    """Test GET /models returns registered model versions and stages."""
    with patch("api.main.get_mlflow_client") as mock_client_getter:
        mock_client = MagicMock()

        mock_version = MagicMock()
        mock_version.name = "MNISTClassifier"
        mock_version.version = "1"
        mock_version.current_stage = "Production"
        mock_version.run_id = "run-xyz-100"
        mock_version.status = "READY"
        mock_version.creation_timestamp = 1600000000
        mock_version.last_updated_timestamp = 1600000500
        mock_version.description = "Production model"
        mock_version.tags = {"accuracy": "0.97"}

        mock_client.search_model_versions.return_value = [mock_version]
        mock_client_getter.return_value = mock_client

        response = test_client.get("/models")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "MNISTClassifier"
        assert data[0]["current_stage"] == "Production"
        assert data[0]["version"] == "1"


def test_predict_endpoint_mocked(test_client, sample_image_data):
    """Test POST /predict endpoint inference with mocked MLflow pyfunc model."""
    import numpy as np

    class MockModel:
        def predict(self, data):
            # Returns class 7 for single input
            return np.array([7])

    with patch("api.main.load_model_from_registry") as mock_loader:
        mock_loader.return_value = (MockModel(), "2", "Production")

        payload = {
            "image_data": sample_image_data,
            "model_name": "MNISTClassifier",
            "stage": "Production"
        }
        response = test_client.post("/predict", json=payload)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["predictions"] == [7]
        assert data["model_name"] == "MNISTClassifier"
        assert data["model_version"] == "2"
        assert data["model_stage"] == "Production"
        assert "inference_time_ms" in data
