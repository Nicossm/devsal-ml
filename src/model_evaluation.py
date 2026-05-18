"""Funciones de evaluación: validación cruzada con múltiples métricas y plots diagnósticos.

Usado por ``03_model_evaluation.ipynb``.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

REGRESSION_SCORING = [
    "r2",
    "neg_mean_absolute_error",
    "neg_root_mean_squared_error",
    "neg_mean_absolute_percentage_error",
]

CLASSIFICATION_SCORING = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

# Renombre sklearn -> nombre humano. Las metricas en _REG_INVERT se reportan positivas (vienen como neg_*).
_REG_RENAME = {
    "r2": "R2",
    "neg_mean_absolute_error": "MAE",
    "neg_root_mean_squared_error": "RMSE",
    "neg_mean_absolute_percentage_error": "MAPE",
}
_REG_INVERT = {"MAE", "RMSE", "MAPE"}

_CLF_RENAME = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC_AUC",
}


def _summarize_cv(
    cv_result: dict[str, np.ndarray],
    scoring: list[str],
    rename: dict[str, str],
    invert: set[str] | None = None,
) -> dict[str, Any]:
    """Convierte la salida de ``cross_validate`` en dict con medias y std por métrica.

    Returns:
        Dict con sub-dicts ``train``/``test`` (claves ``<metrica>_mean``/``_std``)
        más ``fit_time_mean`` y ``score_time_mean`` a nivel superior.
    """
    invert = invert or set()
    summary: dict[str, Any] = {"train": {}, "test": {}}
    for sk_name in scoring:
        nice = rename[sk_name]
        for split in ("train", "test"):
            arr = cv_result[f"{split}_{sk_name}"]
            if nice in invert:
                arr = -arr
            summary[split][f"{nice}_mean"] = float(np.mean(arr))
            summary[split][f"{nice}_std"] = float(np.std(arr))
    summary["fit_time_mean"] = float(np.mean(cv_result["fit_time"]))
    summary["score_time_mean"] = float(np.mean(cv_result["score_time"]))
    return summary


def cross_validate_regression(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """CV de regresión con KFold(shuffle=True): R2, MAE, RMSE y MAPE en train y test."""
    splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_result = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring=REGRESSION_SCORING,
        return_train_score=True,
        n_jobs=-1,
    )
    return _summarize_cv(cv_result, REGRESSION_SCORING, _REG_RENAME, _REG_INVERT)


def cross_validate_classification(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """CV de clasificación binaria con StratifiedKFold: Accuracy, Precision, Recall, F1 y ROC-AUC."""
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    cv_result = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring=CLASSIFICATION_SCORING,
        return_train_score=True,
        n_jobs=-1,
    )
    return _summarize_cv(cv_result, CLASSIFICATION_SCORING, _CLF_RENAME)


def build_comparison_table(
    cv_results_dict: dict[str, dict[str, Any]],
    sort_by: str,
    ascending: bool = False,
) -> pd.DataFrame:
    """Aplana resultados de varios CV en una tabla con columnas ``<metrica>_(mean|std)_(train|test)`` y ``gap_<metrica>``.

    Raises:
        ValueError: Si ``cv_results_dict`` está vacío o ``sort_by`` no es columna del DataFrame.
    """
    if not cv_results_dict:
        raise ValueError("cv_results_dict no puede estar vacio.")

    rows = []
    for name, res in cv_results_dict.items():
        row: dict[str, Any] = {"Modelo": name}
        for split in ("train", "test"):
            for key, val in res[split].items():
                row[f"{key}_{split}"] = val
        for key in res["train"]:
            if key.endswith("_mean"):
                metric = key[: -len("_mean")]
                row[f"gap_{metric}"] = res["train"][key] - res["test"][key]
        row["fit_time_mean"] = res["fit_time_mean"]
        row["score_time_mean"] = res["score_time_mean"]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Modelo")
    if sort_by not in df.columns:
        raise ValueError(
            f"sort_by='{sort_by}' no esta en las columnas del DataFrame. "
            f"Opciones: {list(df.columns)}"
        )
    return df.sort_values(sort_by, ascending=ascending)


def plot_confusion_matrices(
    models_dict: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Grid de matrices de confusión (una por modelo), con valores anotados."""
    n = len(models_dict)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (name, model) in zip(axes_flat, models_dict.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0 (<=med)", "1 (>med)"])
        ax.set_yticklabels(["0 (<=med)", "1 (>med)"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Matrices de confusion en test", fontsize=14, y=1.00)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    models_dict: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple[int, int] = (9, 7),
) -> plt.Figure:
    """Curvas ROC superpuestas de varios modelos, con AUC en la leyenda.

    Raises:
        AttributeError: Si algún pipeline no implementa ``predict_proba``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"El modelo '{name}' no expone predict_proba; no se puede calcular ROC."
            )
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        lw=1,
        label="Aleatorio (AUC = 0.5)",
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
    ax.set_title("Curvas ROC sobre test")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(
    models_dict: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    figsize: tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Grid de gráficos de residuos (y_pred vs residual) para diagnosticar mal-especificación."""
    n = len(models_dict)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, (name, model) in zip(axes_flat, models_dict.items()):
        y_pred = model.predict(X_test)
        residuals = y_test.values - y_pred
        ax.scatter(y_pred, residuals, alpha=0.4, s=15, color="steelblue")
        ax.axhline(0, color="red", linestyle="--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("Prediccion (USD)")
        ax.set_ylabel("Residuo (real - prediccion)")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Graficos de residuos en test", fontsize=14, y=1.00)
    fig.tight_layout()
    return fig
