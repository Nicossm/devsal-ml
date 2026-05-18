"""Modulo de optimizacion de hiperparametros.

Funciones reutilizables para correr ``GridSearchCV`` y
``RandomizedSearchCV`` sobre pipelines supervisados de sklearn, y para
construir tablas comparativas pre/post tuning.

Usado por el notebook ``04_hyperparameter_optimization.ipynb``.

La logica de tuning vive aqui para mantener el notebook como capa de
orquestacion, justificacion de grids y visualizacion -- no de
implementacion.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline


def _make_splitter(
    stratified: bool, cv: int, random_state: int
) -> KFold | StratifiedKFold:
    """Devuelve el splitter adecuado segun la tarea.

    Para clasificacion se usa ``StratifiedKFold`` para mantener el balance
    de clases en cada fold; para regresion se usa ``KFold`` con shuffle.

    Args:
        stratified: ``True`` para clasificacion, ``False`` para regresion.
        cv: Numero de folds.
        random_state: Semilla del shuffle.

    Returns:
        Instancia de ``KFold`` o ``StratifiedKFold`` lista para usar.
    """
    if stratified:
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    return KFold(n_splits=cv, shuffle=True, random_state=random_state)


def run_grid_search(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "r2",
    stratified: bool = False,
    random_state: int = 42,
    verbose: int = 0,
) -> dict[str, Any]:
    """Corre ``GridSearchCV`` sobre un pipeline supervisado.

    Args:
        pipeline: Pipeline de sklearn (preprocesador + estimador). Las
            claves de ``param_grid`` deben llevar el prefijo del paso
            (tipicamente ``"model__"``).
        param_grid: Diccionario de hiperparametros a explorar
            exhaustivamente.
        X: Features de entrenamiento.
        y: Target (numerico para regresion, binario para clasificacion).
        cv: Numero de folds.
        scoring: Metrica de seleccion (ej. ``"r2"``, ``"f1"``).
        stratified: ``True`` si la tarea es clasificacion; controla el
            splitter usado en CV.
        random_state: Semilla del shuffle del splitter.
        verbose: Nivel de verbosidad de ``GridSearchCV``.

    Returns:
        Dict con claves:
        - ``method``: ``"GridSearchCV"``.
        - ``best_estimator``: pipeline reentrenado con los mejores params.
        - ``best_params``: dict de hiperparametros ganadores.
        - ``best_score``: score medio de CV del mejor estimador.
        - ``best_score_std``: desviacion estandar entre folds del ganador.
        - ``n_candidates``: numero total de combinaciones evaluadas.
        - ``cv_results``: ``cv_results_`` resumido a DataFrame
          (ordenado descendente por ``mean_test_score``).
        - ``fit_time_total``: tiempo total acumulado de ajuste (segundos).
    """
    splitter = _make_splitter(stratified, cv, random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=splitter,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=verbose,
    )
    search.fit(X, y)

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    best_idx = search.best_index_
    best_std = float(search.cv_results_["std_test_score"][best_idx])

    return {
        "method": "GridSearchCV",
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_score_std": best_std,
        "n_candidates": int(len(cv_results)),
        "cv_results": cv_results,
        "fit_time_total": float(cv_results["mean_fit_time"].sum() * cv),
    }


def run_randomized_search(
    pipeline: Pipeline,
    param_distributions: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
    scoring: str = "r2",
    stratified: bool = False,
    random_state: int = 42,
    verbose: int = 0,
) -> dict[str, Any]:
    """Corre ``RandomizedSearchCV`` sobre un pipeline supervisado.

    Util cuando el espacio de hiperparametros es grande y un grid
    exhaustivo seria prohibitivo en tiempo.

    Args:
        pipeline: Pipeline de sklearn (preprocesador + estimador). Claves
            de ``param_distributions`` con prefijo del paso (ej. ``"model__"``).
        param_distributions: Diccionario de hiperparametros a muestrear.
            Acepta listas (uniforme discreto) o distribuciones de scipy.
        X: Features de entrenamiento.
        y: Target.
        n_iter: Numero de combinaciones muestreadas.
        cv: Numero de folds.
        scoring: Metrica de seleccion.
        stratified: ``True`` para clasificacion.
        random_state: Semilla -- controla tanto el shuffle del splitter
            como el muestreo de combinaciones.
        verbose: Nivel de verbosidad.

    Returns:
        Mismo formato que ``run_grid_search`` pero con
        ``method="RandomizedSearchCV"`` y ``n_candidates == n_iter``.
    """
    splitter = _make_splitter(stratified, cv, random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=splitter,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        random_state=random_state,
        verbose=verbose,
    )
    search.fit(X, y)

    cv_results = pd.DataFrame(search.cv_results_).sort_values(
        "mean_test_score", ascending=False
    )
    best_idx = search.best_index_
    best_std = float(search.cv_results_["std_test_score"][best_idx])

    return {
        "method": "RandomizedSearchCV",
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "best_score_std": best_std,
        "n_candidates": int(len(cv_results)),
        "cv_results": cv_results,
        "fit_time_total": float(cv_results["mean_fit_time"].sum() * cv),
    }


def build_tuning_comparison_table(
    tuning_results: dict[str, dict[str, Any]],
    baseline_scores: dict[str, float],
    metric_name: str,
) -> pd.DataFrame:
    """Construye una tabla con baseline vs score post-tuning por modelo.

    Args:
        tuning_results: Mapa ``nombre_modelo -> resultado de
            run_grid_search/run_randomized_search``.
        baseline_scores: Mapa ``nombre_modelo -> score baseline``
            (tipicamente la media de CV del notebook 03 sin tuning).
        metric_name: Nombre de la metrica (ej. ``"R2"`` o ``"F1"``) para
            renombrar columnas.

    Returns:
        DataFrame con columnas ``method``, ``<metric>_baseline``,
        ``<metric>_tuned``, ``<metric>_tuned_std``, ``delta``,
        ``delta_pct``, ``n_candidates``, ``fit_time_s``,
        ``best_params``. Ordenado por ``delta`` descendente.

    Raises:
        ValueError: Si las claves de ``tuning_results`` y
            ``baseline_scores`` no coinciden.
    """
    if set(tuning_results.keys()) != set(baseline_scores.keys()):
        raise ValueError(
            "Las claves de tuning_results y baseline_scores deben coincidir. "
            f"Diferencia: {set(tuning_results.keys()) ^ set(baseline_scores.keys())}"
        )

    rows = []
    for name, res in tuning_results.items():
        baseline = float(baseline_scores[name])
        tuned = res["best_score"]
        delta = tuned - baseline
        delta_pct = (delta / baseline * 100.0) if baseline != 0 else float("nan")
        rows.append(
            {
                "Modelo": name,
                "method": res["method"],
                f"{metric_name}_baseline": baseline,
                f"{metric_name}_tuned": tuned,
                f"{metric_name}_tuned_std": res["best_score_std"],
                "delta": delta,
                "delta_pct": delta_pct,
                "n_candidates": res["n_candidates"],
                "fit_time_s": res["fit_time_total"],
                "best_params": str(res["best_params"]),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("Modelo")
        .sort_values("delta", ascending=False)
    )


def export_cv_results_summary(
    tuning_result: dict[str, Any],
    top_n: int = 10,
) -> pd.DataFrame:
    """Extrae las top-N combinaciones de un resultado de tuning.

    Util para reportar en el notebook el espacio de busqueda explorado.

    Args:
        tuning_result: Salida de ``run_grid_search`` o ``run_randomized_search``.
        top_n: Numero de filas top por ``mean_test_score``.

    Returns:
        DataFrame con columnas ``rank_test_score``, ``mean_test_score``,
        ``std_test_score``, ``mean_train_score``, ``mean_fit_time`` y los
        parametros (``param_*``).
    """
    df = tuning_result["cv_results"].head(top_n).copy()
    param_cols = [c for c in df.columns if c.startswith("param_")]
    keep_cols = (
        ["rank_test_score", "mean_test_score", "std_test_score",
         "mean_train_score", "mean_fit_time"]
        + param_cols
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)
