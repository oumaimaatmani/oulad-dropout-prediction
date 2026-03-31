"""
Model training, evaluation, and persistence utilities.

Supports both the XGBoost baseline and the PyTorch MLP hybrid model.
Includes cross-validation, metric computation, threshold calibration,
and model serialization.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# PyTorch MLP -- Lazy-loaded class
# ---------------------------------------------------------------------------

# Global cache for the DropoutMLP class (lazy-loaded on first use)
_DropoutMLP_cache = None


def _get_dropout_mlp_class():
    """
    Factory function to lazily load and cache the DropoutMLP class.
    
    This function imports PyTorch only when called, allowing model_utils
    to be imported without requiring PyTorch at module load time.
    The class is cached globally to avoid repeated imports.
    """
    global _DropoutMLP_cache
    
    if _DropoutMLP_cache is not None:
        return _DropoutMLP_cache
    
    # Lazy import of PyTorch (only when this function is called)
    import torch.nn as nn
    
    class DropoutMLP(nn.Module):
        """
        Multi-layer perceptron for dropout prediction.

        Architecture:
            Input -> Linear(128) -> ReLU -> Dropout(0.3)
                  -> Linear(64)  -> ReLU -> Dropout(0.2)
                  -> Linear(1)   -> Sigmoid

        Designed to combine NLP features with tabular engagement signals
        in a unified feedforward architecture.
        """

        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)
    
    _DropoutMLP_cache = DropoutMLP
    return DropoutMLP


# Don't call _get_dropout_mlp_class() here - let __getattr__ handle lazy loading
# DropoutMLP will be loaded only when explicitly requested (via import or access)

def __getattr__(name: str):
    """
    Module-level __getattr__ for lazy loading of DropoutMLP.
    This allows DropoutMLP to be imported without loading PyTorch at module init time.
    """
    if name == 'DropoutMLP':
        return _get_dropout_mlp_class()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# XGBoost Baseline
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict] = None,
) -> Any:
    """
    Train an XGBoost classifier with early stopping.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data and labels.
    X_val, y_val : array-like
        Validation data and labels for early stopping.
    params : dict, optional
        XGBoost parameters. Uses sensible defaults if not provided.

    Returns
    -------
    xgboost.XGBClassifier
        Trained model.
    """
    import xgboost as xgb

    default_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    return model


def cross_validate_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    params: Optional[Dict] = None,
) -> Dict[str, List[float]]:
    """
    Perform stratified k-fold cross-validation with XGBoost.

    Returns
    -------
    dict
        Dictionary with lists of per-fold metrics:
        roc_auc, f1, precision, recall, accuracy.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = {
        "roc_auc": [], "f1": [], "precision": [],
        "recall": [], "accuracy": [],
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_xgboost(X_train, y_train, X_val, y_val, params)
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results["roc_auc"].append(roc_auc_score(y_val, y_prob))
        results["f1"].append(f1_score(y_val, y_pred))
        results["precision"].append(precision_score(y_val, y_pred))
        results["recall"].append(recall_score(y_val, y_pred))
        results["accuracy"].append(accuracy_score(y_val, y_pred))

        print(f"  Fold {fold}/{n_splits} -- "
              f"AUC: {results['roc_auc'][-1]:.4f}, "
              f"F1: {results['f1'][-1]:.4f}")

    print(f"\nMean ROC-AUC: {np.mean(results['roc_auc']):.4f} "
          f"(+/- {np.std(results['roc_auc']):.4f})")
    print(f"Mean F1:      {np.mean(results['f1']):.4f} "
          f"(+/- {np.std(results['f1']):.4f})")

    return results


# ---------------------------------------------------------------------------
# MLP Training and Prediction
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple:
    """
    Train the DropoutMLP model.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Binary labels (0 or 1).
    input_dim : int
        Number of input features.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    learning_rate : float
        Adam optimizer learning rate.
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple
        (trained_model, list_of_epoch_losses)
    """
    # Lazy import of PyTorch (only when this function is called)
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get the cached DropoutMLP class
    DropoutMLP_Class = _get_dropout_mlp_class()
    model = DropoutMLP_Class(input_dim=input_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{epochs} -- Loss: {avg_loss:.4f}")

    return model, losses


def predict_mlp(
    model, X: np.ndarray, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from a trained MLP.

    Returns
    -------
    tuple
        (probabilities, binary_predictions)
    """
    import torch
    
    model.eval()
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        probs = model(X_tensor).numpy().flatten()

    preds = (probs >= threshold).astype(int)
    return probs, preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute and print classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    print(f"\n{'=' * 50}")
    print(f"  {model_name} -- Evaluation Results")
    print(f"{'=' * 50}")

    for name, value in metrics.items():
        print(f"  {name:>12s}: {value:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Completed", "Dropout"]))

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Compute and return confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Find the classification threshold that maximizes F1-score.

    Returns
    -------
    tuple
        (optimal_threshold, best_f1_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    print(f"Optimal threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return float(best_threshold), float(best_f1)


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_xgboost_model(model: Any, path: str) -> None:
    """Save XGBoost model to native binary format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"XGBoost model saved: {path}")


def load_xgboost_model(path: str) -> Any:
    """Load XGBoost model from file."""
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(path)
    print(f"XGBoost model loaded: {path}")
    return model


def save_mlp_model(
    model, path: str, metadata: Optional[Dict] = None
) -> None:
    """
    Save PyTorch MLP model state dict and optional metadata.
    """
    import torch
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"MLP model saved: {path}")

    if metadata:
        meta_path = path.replace(".pth", "_metadata.json")
        with open(meta_path, "w") as f:
            clean_meta = {}
            for k, v in metadata.items():
                if isinstance(v, np.floating):
                    clean_meta[k] = float(v)
                elif isinstance(v, np.integer):
                    clean_meta[k] = int(v)
                elif isinstance(v, np.ndarray):
                    clean_meta[k] = v.tolist()
                else:
                    clean_meta[k] = v
            json.dump(clean_meta, f, indent=2)
        print(f"Metadata saved: {meta_path}")


def load_mlp_model(path: str, input_dim: int):
    """Load PyTorch MLP model from state dict."""
    import torch
    
    # Get the cached DropoutMLP class
    DropoutMLP_Class = _get_dropout_mlp_class()
    model = DropoutMLP_Class(input_dim=input_dim)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    print(f"MLP model loaded: {path}")
    return model