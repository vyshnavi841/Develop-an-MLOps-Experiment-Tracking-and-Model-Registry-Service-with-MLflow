import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, Any, List, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Local import
try:
    from ml_core.train import train, set_seed
except ImportError:
    from train import train, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ml_core.run_experiments")


# Predefined hyperparameter grid exploring at least 10 distinct configurations
EXPERIMENT_CONFIGURATIONS: List[Dict[str, Any]] = [
    # 1. Random Forest - Baseline
    {
        "model_type": "random_forest",
        "hyperparams": {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2, "criterion": "gini"},
        "run_name": "rf_trees50_depth10"
    },
    # 2. Random Forest - Deep
    {
        "model_type": "random_forest",
        "hyperparams": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 2, "criterion": "gini"},
        "run_name": "rf_trees100_depth20"
    },
    # 3. Random Forest - Large Ensemble
    {
        "model_type": "random_forest",
        "hyperparams": {"n_estimators": 150, "max_depth": 25, "min_samples_split": 4, "criterion": "entropy"},
        "run_name": "rf_trees150_entropy"
    },
    # 4. Random Forest - Regularized
    {
        "model_type": "random_forest",
        "hyperparams": {"n_estimators": 80, "max_depth": 12, "min_samples_split": 6, "criterion": "gini"},
        "run_name": "rf_trees80_regularized"
    },
    # 5. Logistic Regression - Standard
    {
        "model_type": "logistic_regression",
        "hyperparams": {"C": 1.0, "max_iter": 150, "solver": "lbfgs"},
        "run_name": "logreg_c1.0_lbfgs"
    },
    # 6. Logistic Regression - Low Regularization
    {
        "model_type": "logistic_regression",
        "hyperparams": {"C": 0.1, "max_iter": 200, "solver": "lbfgs"},
        "run_name": "logreg_c0.1_lbfgs"
    },
    # 7. Logistic Regression - High C
    {
        "model_type": "logistic_regression",
        "hyperparams": {"C": 5.0, "max_iter": 200, "solver": "lbfgs"},
        "run_name": "logreg_c5.0_lbfgs"
    },
    # 8. Gradient Boosting - Shallow
    {
        "model_type": "gradient_boosting",
        "hyperparams": {"n_estimators": 40, "learning_rate": 0.1, "max_depth": 3},
        "run_name": "gbm_lr0.1_depth3"
    },
    # 9. Gradient Boosting - Conservative
    {
        "model_type": "gradient_boosting",
        "hyperparams": {"n_estimators": 60, "learning_rate": 0.05, "max_depth": 4},
        "run_name": "gbm_lr0.05_depth4"
    },
    # 10. Gradient Boosting - Aggressive
    {
        "model_type": "gradient_boosting",
        "hyperparams": {"n_estimators": 80, "learning_rate": 0.2, "max_depth": 3},
        "run_name": "gbm_lr0.2_depth3"
    },
    # 11. Random Forest - Tuned Final
    {
        "model_type": "random_forest",
        "hyperparams": {"n_estimators": 120, "max_depth": 18, "min_samples_split": 3, "criterion": "gini"},
        "run_name": "rf_trees120_depth18"
    }
]


def run_experiment_sweep(
    experiment_name: str = "MNIST-Classification",
    num_runs: int = 10,
    subset_size: int = 4000,
    random_seed: int = 42,
    tracking_uri: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Execute hyperparameter sweep generating distinct runs logged to MLflow.
    """
    logger.info(f"Starting sweep of {num_runs} experiments for '{experiment_name}'...")
    set_seed(random_seed)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    results = []
    configs_to_run = EXPERIMENT_CONFIGURATIONS[:num_runs]

    for idx, config in enumerate(configs_to_run, 1):
        logger.info(f"\n[{idx}/{num_runs}] Running Config: {config['run_name']} ({config['model_type']})")
        try:
            res = train(
                experiment_name=experiment_name,
                run_name=config["run_name"],
                model_type=config["model_type"],
                hyperparams=config["hyperparams"],
                subset_size=subset_size,
                random_state=random_seed + idx,
                tracking_uri=tracking_uri
            )
            results.append(res)
        except Exception as e:
            logger.error(f"Error executing run {config['run_name']}: {e}", exc_info=True)

    return results


def find_best_run_and_register(
    experiment_name: str = "MNIST-Classification",
    model_name: str = "MNISTClassifier",
    metric_name: str = "accuracy",
    target_stage: str = "Staging",
    tracking_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query MLflow Tracking for the top performing run and register it to MLflow Model Registry.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found!")

    logger.info(f"Searching runs for experiment '{experiment_name}' (ID: {experiment.experiment_id}) sorted by {metric_name} DESC...")

    # Search runs sorted by target metric descending
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=20
    )

    if not runs:
        raise RuntimeError(f"No runs found for experiment '{experiment_name}'!")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_metric_val = best_run.data.metrics.get(metric_name, 0.0)
    best_model_type = best_run.data.params.get("model_type", "unknown")

    logger.info("=" * 60)
    logger.info(f"🏆 BEST RUN IDENTIFIED:")
    logger.info(f"   Run ID:       {best_run_id}")
    logger.info(f"   Run Name:     {best_run.data.tags.get('mlflow.runName', 'Unnamed')}")
    logger.info(f"   {metric_name.upper()}:     {best_metric_val:.4f}")
    logger.info(f"   Model Type:   {best_model_type}")
    logger.info(f"   Parameters:   {json.dumps(best_run.data.params, indent=2)}")
    logger.info("=" * 60)

    # Register Model in MLflow Model Registry
    model_source = f"runs:/{best_run_id}/model"
    logger.info(f"Registering model '{model_name}' from source: {model_source}")

    try:
        # Create registered model if it does not exist
        try:
            client.create_registered_model(
                name=model_name,
                description="MNIST digit classification model registered and governed via MLOps pipeline."
            )
            logger.info(f"Created new registered model '{model_name}'")
        except Exception:
            logger.info(f"Registered model '{model_name}' already exists in registry.")

        # Create model version
        model_version = client.create_model_version(
            name=model_name,
            source=model_source,
            run_id=best_run_id,
            description=f"Model trained with {best_model_type}, {metric_name}={best_metric_val:.4f}"
        )
        logger.info(f"Created Model Version: {model_version.version}")

        # Transition stage
        if target_stage:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=target_stage,
                archive_existing_versions=False
            )
            logger.info(f"Transitioned Model Version {model_version.version} to Stage: '{target_stage}'")

        # Set model alias (MLflow 2.x alias support)
        try:
            alias_name = "champion" if target_stage.lower() == "production" else "challenger"
            client.set_registered_model_alias(
                name=model_name,
                alias=alias_name,
                version=model_version.version
            )
            logger.info(f"Assigned alias '{alias_name}' to version {model_version.version}")
        except Exception as e:
            logger.debug(f"Alias assignment note: {e}")

        summary = {
            "status": "SUCCESS",
            "model_name": model_name,
            "version": model_version.version,
            "stage": target_stage,
            "best_run_id": best_run_id,
            "primary_metric": metric_name,
            "best_metric_value": best_metric_val,
            "model_type": best_model_type,
            "parameters": best_run.data.params
        }
        return summary

    except Exception as e:
        logger.error(f"Failed to register model: {e}", exc_info=True)
        raise e


def parse_args():
    parser = argparse.ArgumentParser(description="Orchestrate hyperparameter tuning sweep and register champion model.")
    parser.add_argument("--experiment_name", type=str, default="MNIST-Classification", help="MLflow experiment name")
    parser.add_argument("--model_name", type=str, default="MNISTClassifier", help="Registered model name")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of distinct runs (min 10)")
    parser.add_argument("--subset_size", type=int, default=4000, help="Subset size for training")
    parser.add_argument("--metric", type=str, default="accuracy", help="Primary evaluation metric")
    parser.add_argument("--stage", type=str, default="Staging", choices=["Staging", "Production", "Archived"])
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--tracking_uri", type=str, default=None, help="MLflow tracking URI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("================ STARTING EXPERIMENT SUITE ================")
    sweep_results = run_experiment_sweep(
        experiment_name=args.experiment_name,
        num_runs=args.num_runs,
        subset_size=args.subset_size,
        random_seed=args.seed,
        tracking_uri=args.tracking_uri
    )
    logger.info(f"Completed {len(sweep_results)} runs. Selecting best model...")

    best_registration = find_best_run_and_register(
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        metric_name=args.metric,
        target_stage=args.stage,
        tracking_uri=args.tracking_uri
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT ORCHESTRATION & REGISTRY SUMMARY:")
    print(json.dumps(best_registration, indent=2))
    print("=" * 60 + "\n")
