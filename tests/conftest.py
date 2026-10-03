import os
import sys
import tempfile
import pytest
from fastapi.testclient import TestClient
import numpy as np
import mlflow

# Add project root to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app


@pytest.fixture(scope="session")
def test_client():
    """FastAPI Test Client fixture."""
    return TestClient(app)


@pytest.fixture(scope="session")
def sample_image_data():
    """Generates a sample 784-length normalized digit pixel list."""
    np.random.seed(42)
    # A simple digit mock pattern
    arr = np.zeros(784, dtype=float)
    arr[200:250] = 0.8
    arr[300:350] = 0.95
    return arr.tolist()


@pytest.fixture(scope="session")
def sample_batch_images():
    """Generates a batch of 3 sample 784-length images."""
    np.random.seed(42)
    return [np.random.rand(784).tolist() for _ in range(3)]


@pytest.fixture(scope="session")
def temp_mlflow_environment():
    """Sets up a temporary SQLite/local file backend for local integration testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_mlflow.db")
    mlruns_path = os.path.join(temp_dir, "test_mlruns")
    os.makedirs(mlruns_path, exist_ok=True)

    sqlite_uri = f"sqlite:///{db_path}"
    os.environ["MLFLOW_TRACKING_URI"] = sqlite_uri
    mlflow.set_tracking_uri(sqlite_uri)

    yield {"tracking_uri": sqlite_uri, "artifact_dir": mlruns_path}
