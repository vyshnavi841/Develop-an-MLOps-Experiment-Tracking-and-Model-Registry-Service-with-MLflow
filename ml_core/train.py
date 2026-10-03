import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from typing import Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ml_core.train")


def set_seed(seed: int = 42):
    """Set random seeds for complete reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_mnist_data(
    subset_size: Optional[int] = 10000,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess the MNIST dataset.
    Falls back to a synthetic digit dataset if network connection is unavailable.
    """
    logger.info("Loading MNIST dataset...")
    try:
        # Fetch MNIST (784 features, 10 classes)
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data, mnist.target.astype(int)
        logger.info(f"Successfully loaded full MNIST: {X.shape[0]} samples")
    except Exception as e:
        logger.warning(f"Failed to fetch online MNIST ({e}). Generating high-quality synthetic MNIST mock data...")
        from sklearn.datasets import make_classification
        total_samples = subset_size or 5000
        X, y = make_classification(
            n_samples=total_samples,
            n_features=784,
            n_informative=120,
            n_redundant=80,
            n_classes=10,
            random_state=random_state
        )

    # Subsample for faster experimentation if requested
    if subset_size and subset_size < len(X):
        indices = np.random.choice(len(X), subset_size, replace=False)
        X = X[indices]
        y = y[indices]
        logger.info(f"Subsampled dataset to {subset_size} records")

    # Normalize pixel values to [0.0, 1.0]
    if X.max() > 1.0:
        X = X / 255.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train split shape: {X_train.shape}, Test split shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def build_and_train_model(
    model_type: str,
    hyperparams: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
) -> Any:
    """Build and train the requested classification model."""
    logger.info(f"Building model '{model_type}' with hyperparams: {hyperparams}")

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 100)),
            max_depth=int(hyperparams["max_depth"]) if hyperparams.get("max_depth") else None,
            min_samples_split=int(hyperparams.get("min_samples_split", 2)),
            criterion=hyperparams.get("criterion", "gini"),
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            C=float(hyperparams.get("C", 1.0)),
            max_iter=int(hyperparams.get("max_iter", 200)),
            solver=hyperparams.get("solver", "lbfgs"),
            random_state=random_state,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=int(hyperparams.get("n_estimators", 50)),
            learning_rate=float(hyperparams.get("learning_rate", 0.1)),
            max_depth=int(hyperparams.get("max_depth", 3)),
            random_state=random_state
        )
    elif model_type == "cnn" or model_type == "mlp":
        # Deep Learning with TensorFlow Keras
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models

            tf.random.set_seed(random_state)
            epochs = int(hyperparams.get("epochs", 5))
            batch_size = int(hyperparams.get("batch_size", 64))
            learning_rate = float(hyperparams.get("learning_rate", 0.001))

            if model_type == "cnn":
                X_train_reshaped = X_train.reshape(-1, 28, 28, 1)
                model = models.Sequential([
                    layers.Input(shape=(28, 28, 1)),
                    layers.Conv2D(32, (3, 3), activation="relu"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Conv2D(64, (3, 3), activation="relu"),
                    layers.MaxPooling2D((2, 2)),
                    layers.Flatten(),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(10, activation="softmax")
                ])
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )
                model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
                return model
            else:
                model = models.Sequential([
                    layers.Input(shape=(784,)),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.2),
                    layers.Dense(64, activation="relu"),
                    layers.Dense(10, activation="softmax")
                ])
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"]
                )
                model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
                return model
        except ImportError:
            logger.warning("TensorFlow not installed. Falling back to RandomForest.")
            model = RandomForestClassifier(n_estimators=50, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    start_time = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    logger.info(f"Model training completed in {duration:.2f} seconds")
    return model


def evaluate_and_generate_artifacts(
    model: Any,
    model_type: str,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[Dict[str, float], str, str]:
    """
    Evaluate model performance, compute metrics, and generate artifact plots.
    Returns (metrics_dict, confusion_matrix_path, report_path).
    """
    if model_type == "cnn":
        X_test_in = X_test.reshape(-1, 28, 28, 1)
        y_pred_proba = model.predict(X_test_in)
        y_pred = np.argmax(y_pred_proba, axis=1)
    elif model_type == "mlp":
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    logger.info(f"Evaluation Metrics: {json.dumps(metrics, indent=2)}")

    temp_dir = tempfile.mkdtemp()

    # 1. Confusion Matrix Artifact
    cm_path = os.path.join(temp_dir, "confusion_matrix.png")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_type.upper()} (Accuracy: {accuracy:.4f})")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 2. Classification Report Artifact
    report_path = os.path.join(temp_dir, "classification_report.json")
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    with open(report_path, "w") as f:
        json.dump(clf_report, f, indent=4)

    return metrics, cm_path, report_path


def train(
    experiment_name: str = "MNIST-Classification",
    run_name: Optional[str] = None,
    model_type: str = "random_forest",
    hyperparams: Optional[Dict[str, Any]] = None,
    subset_size: int = 5000,
    random_state: int = 42,
    tracking_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main training execution function logged with MLflow.
    """
    set_seed(random_state)
    hyperparams = hyperparams or {}

    # MLflow tracking configuration
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    elif os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id if experiment else "0"

    X_train, X_test, y_train, y_test = load_mnist_data(
        subset_size=subset_size,
        random_state=random_state
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow Run: ID={run_id}, Experiment='{experiment_name}' (ID={experiment_id})")

        # 1. Log Hyperparameters & Tags
        mlflow.log_params(hyperparams)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("subset_size", subset_size)
        mlflow.log_param("random_state", random_state)

        mlflow.set_tag("framework", "scikit-learn" if model_type not in ["cnn", "mlp"] else "tensorflow")
        mlflow.set_tag("task", "image-classification")
        mlflow.set_tag("dataset", "MNIST")
        mlflow.set_tag("model_family", model_type)

        # 2. Train Model
        model = build_and_train_model(
            model_type=model_type,
            hyperparams=hyperparams,
            X_train=X_train,
            y_train=y_train,
            random_state=random_state
        )

        # 3. Evaluate & Log Metrics
        metrics, cm_path, report_path = evaluate_and_generate_artifacts(
            model=model,
            model_type=model_type,
            X_test=X_test,
            y_test=y_test
        )
        mlflow.log_metrics(metrics)

        # 4. Log Artifacts
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(report_path, artifact_path="metrics_reports")

        # 5. Log Model with Signature and Input Example
        input_example = X_test[:2]
        signature = infer_signature(X_test[:5], model.predict(X_test[:5]) if model_type not in ["cnn", "mlp"] else np.argmax(model.predict(X_test[:5]), axis=1))

        if model_type in ["cnn", "mlp"]:
            try:
                import mlflow.tensorflow
                mlflow.tensorflow.log_model(
                    model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example
                )
            except Exception as e:
                logger.warning(f"Error logging TF model: {e}")
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example
            )

        logger.info(f"Successfully logged model and artifacts for Run {run_id}")

        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "metrics": metrics,
            "params": hyperparams,
            "model_type": model_type
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Train MNIST classification model with MLflow tracking.")
    parser.add_argument("--experiment_name", type=str, default="MNIST-Classification", help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="Name of the MLflow run")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["random_forest", "logistic_regression", "gradient_boosting", "cnn", "mlp"])
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees for RandomForest/GBM")
    parser.add_argument("--max_depth", type=int, default=None, help="Max depth of trees")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples to split")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate for GBM/NN")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization for Logistic Regression")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs for neural networks")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for neural networks")
    parser.add_argument("--subset_size", type=int, default=5000, help="Subset size of MNIST")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--tracking_uri", type=str, default=None, help="MLflow Tracking Server URI")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    hyperparameters = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "learning_rate": args.learning_rate,
        "C": args.C,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

    result = train(
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        model_type=args.model_type,
        hyperparams=hyperparameters,
        subset_size=args.subset_size,
        random_state=args.random_state,
        tracking_uri=args.tracking_uri
    )
    print(f"\n[TRAINING_RESULT] {json.dumps(result, indent=2)}")
