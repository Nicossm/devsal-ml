"""Funciones para correr GridSearchCV / RandomizedSearchCV y comparar resultados pre/post tuning.

Usado por ``04_hyperparameter_optimization.ipynb``.
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
    """StratifiedKFold si stratified=True (clasificación), KFold con shuffle en caso contrario."""
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
    """Corre ``GridSearchCV`` sobre un pipeline. Claves de ``param_grid`` con prefijo del paso (ej. ``model__``).

    Returns:
        Dict con ``method``, ``best_estimator``, ``best_params``, ``best_score``,
        ``best_score_std``, ``n_candidates``, ``cv_results`` (DataFrame ordenado)
        y ``fit_time_total`` en segundos.
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
    """Corre ``RandomizedSearchCV`` (acepta listas o distribuciones scipy en ``param_distributions``).

    Returns:
        Mismo formato que :func:`run_grid_search` pero con ``method="RandomizedSearchCV"``
        y ``n_candidates == n_iter``.
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
    """Tabla con baseline vs score post-tuning por modelo, con delta absoluto y porcentual.

    Returns:
        DataFrame con ``method``, ``<metric>_baseline``, ``<metric>_tuned``,
        ``<metric>_tuned_std``, ``delta``, ``delta_pct``, ``n_candidates``,
        ``fit_time_s`` y ``best_params``. Ordenado por ``delta`` desc.

    Raises:
        ValueError: Si las claves de ``tuning_results`` y ``baseline_scores`` no coinciden.
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
    """Top-N combinaciones del ``cv_results_`` ordenadas por ``mean_test_score``."""
    df = tuning_result["cv_results"].head(top_n).copy()
    param_cols = [c for c in df.columns if c.startswith("param_")]
    keep_cols = (
        ["rank_test_score", "mean_test_score", "std_test_score",
         "mean_train_score", "mean_fit_time"]
        + param_cols
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)
