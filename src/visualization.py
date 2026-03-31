"""
Visualization and reporting utilities for OULAD dropout prediction.

Provides consistent, publication-quality plots for:
- Data exploration (distributions, correlations)
- Model evaluation (ROC curves, confusion matrices, PR curves)
- Feature importance (SHAP, XGBoost native)
- Training diagnostics (loss curves)
- Comparative analysis (model benchmarks)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Style Configuration
# ---------------------------------------------------------------------------

STYLE_CONFIG = {
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

COLORS = {
    "primary": "#2C6E91",
    "secondary": "#E8734A",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "danger": "#DC3545",
    "neutral": "#6C757D",
    "dropout": "#DC3545",
    "completed": "#4CAF50",
    "accent": "#7B68EE",
}


def apply_style():
    """Apply consistent plot styling."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette([
        COLORS["primary"], COLORS["secondary"], COLORS["success"],
        COLORS["warning"], COLORS["accent"], COLORS["neutral"],
    ])


def _save_figure(fig: plt.Figure, path: str) -> None:
    """Save figure with tight layout."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Figure saved: {path}")


# ---------------------------------------------------------------------------
# Data Exploration Plots
# ---------------------------------------------------------------------------

def plot_target_distribution(
    y: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot distribution of dropout vs completed students."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = y.value_counts().sort_index()
    labels = ["Completed", "Dropout"]
    colors = [COLORS["completed"], COLORS["dropout"]]

    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, counts.values):
        pct = count / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.02,
            f"{count:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Number of Students")
    ax.set_title("Student Outcome Distribution")
    ax.set_ylim(0, counts.max() * 1.2)

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "is_dropout",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot distributions of selected features, split by target."""
    apply_style()
    n = len(features)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        for label, color, name in [
            (0, COLORS["completed"], "Completed"),
            (1, COLORS["dropout"], "Dropout"),
        ]:
            subset = df[df[target_col] == label][feat].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=name, density=True)
        ax.set_xlabel(feat)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Distributions by Outcome", fontsize=14, y=1.02)

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot correlation heatmap for selected features."""
    apply_style()

    if features:
        data = df[features]
    else:
        data = df.select_dtypes(include=[np.number])

    corr = data.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Matrix")

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Model Evaluation Plots
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot ROC curve with AUC score."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=2.5,
            label=f"{model_name} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_roc_comparison(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same axes.

    Parameters
    ----------
    results : dict
        Mapping of model_name -> (y_true, y_prob).
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = list(COLORS.values())

    for i, (name, (y_true, y_prob)) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2.5,
                label=f"{name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot precision-recall curve."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    ax.plot(recall, precision, color=COLORS["secondary"], linewidth=2.5,
            label=f"{model_name} (PR-AUC = {pr_auc:.4f})")

    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.4,
               label=f"Baseline ({baseline:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot annotated confusion matrix."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm / cm.sum() * 100

    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Completed", "Dropout"],
        yticklabels=["Completed", "Dropout"],
        linewidths=1, linecolor="white",
    )

    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)",
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {model_name}")

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Feature Importance Plots
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importance."""
    apply_style()

    top = importance.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    bars = ax.barh(
        range(len(top)), top.values,
        color=COLORS["primary"], edgecolor="white", linewidth=0.5,
    )

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)

    for bar, val in zip(bars, top.values):
        ax.text(
            val + top.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            ha="left", va="center", fontsize=9,
        )

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Training Diagnostics
# ---------------------------------------------------------------------------

def plot_training_loss(
    losses: List[float],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training loss curve across epochs."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        range(1, len(losses) + 1), losses,
        color=COLORS["primary"], linewidth=2.5, marker="o",
        markersize=4, markerfacecolor=COLORS["secondary"],
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_cross_validation_results(
    cv_results: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cross-validation metric distributions."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = list(cv_results.keys())
    positions = range(len(metrics))

    bp = ax.boxplot(
        [cv_results[m] for m in metrics],
        positions=positions,
        widths=0.5,
        patch_artist=True,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["primary"])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("Cross-Validation Results")

    # Add mean values as text
    for i, m in enumerate(metrics):
        mean_val = np.mean(cv_results[m])
        ax.text(i, mean_val + 0.005, f"{mean_val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    if save_path:
        _save_figure(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# NLP-Specific Plots
# ---------------------------------------------------------------------------

def plot_nlp_sentiment_distribution(
    nlp_df: pd.DataFrame,
    target: pd.Series,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot NLP sentiment distribution by dropout status."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sentiment label distribution
    ax = axes[0]
    for label, color, name in [
        (0, COLORS["completed"], "Completed"),
        (1, COLORS["dropout"], "Dropout"),
    ]:
        mask = target == label
        counts = nlp_df.loc[mask, "nlp_sentiment"].value_counts()
        pcts = counts / counts.sum() * 100
        bars = ax.bar(
            [f"{s}\n({name})" for s in pcts.index],
            pcts.values,
            color=color, alpha=0.7, edgecolor="white",
        )
    ax.set_ylabel("Percentage")
    ax.set_title("NLP Sentiment by Outcome")

    # Confidence distribution
    ax = axes[1]
    for label, color, name in [
        (0, COLORS["completed"], "Completed"),
        (1, COLORS["dropout"], "Dropout"),
    ]:
        mask = target == label
        ax.hist(
            nlp_df.loc[mask, "nlp_confidence"],
            bins=30, alpha=0.6, color=color, label=name, density=True,
        )
    ax.set_xlabel("NLP Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("Sentiment Confidence Distribution")
    ax.legend()

    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_model_comparison_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a visual comparison table of model metrics.

    Parameters
    ----------
    metrics_dict : dict
        Mapping of model_name -> {metric_name: value}.
    """
    apply_style()

    df = pd.DataFrame(metrics_dict).T
    df = df.round(4)

    fig, ax = plt.subplots(figsize=(10, 2 + len(df) * 0.6))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=[c.replace("_", " ").title() for c in df.columns],
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Style header
    for j in range(len(df.columns)):
        table[0, j].set_facecolor(COLORS["primary"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best values per column
    for j, col in enumerate(df.columns):
        best_idx = df[col].idxmax()
        row_idx = list(df.index).index(best_idx) + 1
        table[row_idx, j].set_facecolor("#E8F4FD")

    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=20)

    if save_path:
        _save_figure(fig, save_path)
    return fig
