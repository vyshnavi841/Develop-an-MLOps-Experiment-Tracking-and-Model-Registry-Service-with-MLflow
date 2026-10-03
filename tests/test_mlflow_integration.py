import pytest
import mlflow
from mlflow.tracking import MlflowClient
from ml_core.train import train
from ml_core.run_experiments import find_best_run_and_register


def test_mlflow_single_train_run(temp_mlflow_environment):
    """Integration test: Train run records params, metrics, and model artifact to MLflow."""
    uri = temp_mlflow_environment["tracking_uri"]

    res = train(
        experiment_name="Integration-Test-Experiment",
        run_name="integration_test_run",
        model_type="random_forest",
        hyperparams={"n_estimators": 10, "max_depth": 5},
        subset_size=300,
        random_state=42,
        tracking_uri=uri
    )

    assert res["status"] == "SUCCESS"
    assert res["run_id"] is not None
    assert "accuracy" in res["metrics"]

    client = MlflowClient(tracking_uri=uri)
    run = client.get_run(res["run_id"])
    assert run.info.status == "FINISHED"
    assert "accuracy" in run.data.metrics
    assert run.data.params.get("n_estimators") == "10"


def test_mlflow_model_registration_and_stage_transition(temp_mlflow_environment):
    """Integration test: Best run is identified, registered to Model Registry, and staged."""
    uri = temp_mlflow_environment["tracking_uri"]
    exp_name = "Registry-Test-Experiment"
    model_name = "IntegrationMNISTModel"

    # Log 2 runs
    r1 = train(
        experiment_name=exp_name,
        run_name="run_1",
        model_type="random_forest",
        hyperparams={"n_estimators": 5, "max_depth": 3},
        subset_size=200,
        random_state=1,
        tracking_uri=uri
    )
    r2 = train(
        experiment_name=exp_name,
        run_name="run_2",
        model_type="random_forest",
        hyperparams={"n_estimators": 25, "max_depth": 10},
        subset_size=200,
        random_state=2,
        tracking_uri=uri
    )

    # Register best
    reg_summary = find_best_run_and_register(
        experiment_name=exp_name,
        model_name=model_name,
        metric_name="accuracy",
        target_stage="Staging",
        tracking_uri=uri
    )

    assert reg_summary["status"] == "SUCCESS"
    assert reg_summary["model_name"] == model_name
    assert reg_summary["version"] == "1"
    assert reg_summary["stage"] == "Staging"

    # Verify via MlflowClient
    client = MlflowClient(tracking_uri=uri)
    latest_staging = client.get_latest_versions(model_name, stages=["Staging"])
    assert len(latest_staging) == 1
    assert latest_staging[0].version == "1"
    assert latest_staging[0].current_stage == "Staging"

    # Transition to Production
    client.transition_model_version_stage(
        name=model_name,
        version="1",
        stage="Production"
    )
    prod_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    assert prod_version.current_stage == "Production"
