"""Modulo de evaluacion de modelos supervisados.

Funciones reutilizables para validacion cruzada con multiples metricas y
para generar visualizaciones diagnosticas (matrices de confusion, curvas
ROC, graficos de residuos). Usado por el notebook
``03_model_evaluation.ipynb``.

La logica de evaluacion vive aqui para mantener el notebook como capa
de orquestacion y narracion, no de implementacion.
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

# Mapeos sklearn-scoring -> nombre humano en la tabla final.
# Las metricas "neg_*" se invierten al exportar para reportarlas siempre
# positivas, mas naturales de interpretar.
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
    """Convierte la salida cruda de ``cross_validate`` en un dict de medias y std.

    Args:
        cv_result: Dict retornado por ``cross_validate``.
        scoring: Lista de claves de scoring usadas en la llamada.
        rename: Mapeo sklearn-scoring -> nombre humano.
        invert: Conjunto de nombres ya renombrados cuyo signo hay que invertir
            (para las metricas que vienen como ``neg_*``).

    Returns:
        Dict con dos sub-dicts ``train`` y ``test``; cada uno mapea
        ``"<metrica>_mean"`` y ``"<metrica>_std"`` a un float, mas las
        claves de nivel superior ``fit_time_mean`` y ``score_time_mean``.
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
    """Corre validacion cruzada de regresion con multiples metricas.

    Usa ``KFold(shuffle=True)`` con ``random_state`` fijo y devuelve la
    media y desviacion estandar de R^2, MAE, RMSE y MAPE en train y test.
    El preprocesamiento incluido en el ``Pipeline`` se reaplica en cada
    fold (sin leakage entre folds).

    Args:
        pipeline: Pipeline de regresion (preprocesador + estimador).
        X: Features.
        y: Target numerico.
        cv: Numero de folds.
        random_state: Semilla para el shuffle del KFold.

    Returns:
        Dict con claves ``train``, ``test``, ``fit_time_mean`` y
        ``score_time_mean``. Cada sub-dict ``train``/``test`` contiene
        ``R2_mean``, ``R2_std``, ``MAE_mean``, ``MAE_std``, ``RMSE_mean``,
        ``RMSE_std``, ``MAPE_mean``, ``MAPE_std``.
    """
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
    """Corre validacion cruzada de clasificacion binaria con multiples metricas.

    Usa ``StratifiedKFold(shuffle=True)`` con ``random_state`` fijo y
    devuelve la media y desviacion estandar de Accuracy, Precision,
    Recall, F1 y ROC-AUC en train y test.

    Args:
        pipeline: Pipeline de clasificacion (preprocesador + estimador).
        X: Features.
        y: Target binario.
        cv: Numero de folds.
        random_state: Semilla para el shuffle del StratifiedKFold.

    Returns:
        Dict con claves ``train``, ``test``, ``fit_time_mean`` y
        ``score_time_mean``. Cada sub-dict ``train``/``test`` contiene
        ``<metrica>_mean`` y ``<metrica>_std`` para cada metrica.
    """
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
    """Construye un DataFrame comparativo a partir de varios resultados de CV.

    Aplana el dict anidado producido por ``cross_validate_*`` en una sola
    fila por modelo, con columnas ``<metrica>_(mean|std)_(train|test)``,
    ``gap_<metrica>`` (train_mean - test_mean) y los tiempos.

    Args:
        cv_results_dict: Mapa ``nombre_modelo -> resultado de cross_validate_*``.
        sort_by: Nombre de la columna por la que ordenar (ej. ``"R2_mean_test"``).
        ascending: Direccion del orden.

    Returns:
        DataFrame con un modelo por fila, ordenado segun ``sort_by``.

    Raises:
        ValueError: Si ``cv_results_dict`` esta vacio o ``sort_by`` no es
            una columna del DataFrame resultante.
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
    """Dibuja una grid 2x2 de matrices de confusion para 4 modelos.

    Args:
        models_dict: Mapa ``nombre_modelo -> pipeline_entrenado`` (cargado
            por ejemplo desde ``joblib``). Se esperan 4 modelos para
            llenar la grid 2x2; si hay mas o menos, la grid se ajusta.
        X_test: Features de test.
        y_test: Target binario de test.
        figsize: Tamano de la figura.

    Returns:
        El objeto ``Figure``.
    """
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
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=12)
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
    """Dibuja las curvas ROC de varios modelos superpuestas, con AUC en la leyenda.

    Args:
        models_dict: Mapa ``nombre_modelo -> pipeline_entrenado``. Los
            modelos deben exponer ``predict_proba``.
        X_test: Features de test.
        y_test: Target binario de test.
        figsize: Tamano de la figura.

    Returns:
        El objeto ``Figure``.

    Raises:
        AttributeError: Si algun pipeline no implementa ``predict_proba``.
    """
    fig, ax = plt.subplots(figsize=figsize)
    for name, model in models_dict.items():
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                f"El modelo '{name}' no expone predict_proba; no se puede "
                "calcular ROC."
            )
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1,
            label="Aleatorio (AUC = 0.5)")
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
    """Dibuja una grid 2x2 de graficos de residuos (y_pred vs residual).

    Un residuo bien comportado se distribuye aleatoriamente alrededor de
    cero a lo largo de todo el rango de prediccion. Patrones (curvatura,
    embudo, sesgo) sugieren mal-especificacion del modelo.

    Args:
        models_dict: Mapa ``nombre_modelo -> pipeline_entrenado``.
        X_test: Features de test.
        y_test: Target numerico de test.
        figsize: Tamano de la figura.

    Returns:
        El objeto ``Figure``.
    """
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
