import numpy as np
import pytest
from ml_core.train import load_mnist_data, build_and_train_model, evaluate_and_generate_artifacts


def test_load_mnist_data_shapes():
    """Verify data loader produces correct array dimensions and normalized values."""
    X_train, X_test, y_train, y_test = load_mnist_data(subset_size=200, test_size=0.25, random_state=42)

    assert X_train.shape == (150, 784)
    assert X_test.shape == (50, 784)
    assert len(y_train) == 150
    assert len(y_test) == 50

    # Ensure normalized to [0, 1]
    assert X_train.max() <= 1.0
    assert X_train.min() >= 0.0


def test_build_and_train_random_forest():
    """Verify RandomForest model builds and trains properly on mock subset."""
    X_train, X_test, y_train, y_test = load_mnist_data(subset_size=200, random_state=42)
    model = build_and_train_model(
        model_type="random_forest",
        hyperparams={"n_estimators": 10, "max_depth": 5},
        X_train=X_train,
        y_train=y_train,
        random_state=42
    )

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset(set(range(10)))


def test_build_and_train_logistic_regression():
    """Verify Logistic Regression trains and predicts valid classes."""
    X_train, X_test, y_train, y_test = load_mnist_data(subset_size=200, random_state=42)
    model = build_and_train_model(
        model_type="logistic_regression",
        hyperparams={"C": 0.5, "max_iter": 50},
        X_train=X_train,
        y_train=y_train,
        random_state=42
    )

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)


def test_evaluate_and_generate_artifacts():
    """Verify evaluation generates metrics dictionary and visual artifact paths."""
    import os
    X_train, X_test, y_train, y_test = load_mnist_data(subset_size=200, random_state=42)
    model = build_and_train_model(
        model_type="random_forest",
        hyperparams={"n_estimators": 5},
        X_train=X_train,
        y_train=y_train
    )

    metrics, cm_path, report_path = evaluate_and_generate_artifacts(
        model=model,
        model_type="random_forest",
        X_test=X_test,
        y_test=y_test
    )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    assert os.path.exists(cm_path)
    assert os.path.exists(report_path)
