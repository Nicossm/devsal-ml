"""Funciones reutilizables para pipelines, entrenamiento y serialización de modelos.

Usado por ``02_supervised_modeling.ipynb``.
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EXPECTED_COLUMNS = [
    "experience",
    "country",
    "education",
    "languages",
    "frameworks",
    "company_size",
    "salary_usd",
    "num_languages",
    "num_frameworks",
]


def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV limpio y valida que tenga las columnas esperadas.

    Raises:
        FileNotFoundError: Archivo no encontrado en ``path``.
        ValueError: Faltan columnas requeridas.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No se encontro el archivo de datos en '{path}'. "
            "Verifica la ruta o ejecuta el pipeline ETL."
        ) from exc

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"El dataset carece de columnas requeridas: {missing}. "
            f"Columnas presentes: {list(df.columns)}"
        )
    return df


def create_binary_target(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[pd.Series, pd.Series, float]:
    """Crea target binario usando la mediana de train como umbral (sin leakage).

    Returns:
        Tupla ``(y_train_binary, y_test_binary, threshold)``.

    Raises:
        TypeError: Si los argumentos no son ``pd.Series``.
    """
    if not isinstance(y_train, pd.Series) or not isinstance(y_test, pd.Series):
        raise TypeError("y_train y y_test deben ser pd.Series.")

    threshold = float(y_train.median())
    y_train_binary = (y_train > threshold).astype(int)
    y_test_binary = (y_test > threshold).astype(int)
    return y_train_binary, y_test_binary, threshold


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Crea un ColumnTransformer: StandardScaler para numéricas, OneHotEncoder para categóricas.

    Raises:
        ValueError: Si alguna de las listas de features está vacía.
    """
    if not numeric_features:
        raise ValueError("numeric_features no puede estar vacio.")
    if not categorical_features:
        raise ValueError("categorical_features no puede estar vacio.")

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )


def build_regression_pipeline(model: Any, preprocessor: ColumnTransformer) -> Pipeline:
    """Retorna Pipeline(preprocessor, model) para regresión."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_classification_pipeline(
    model: Any, preprocessor: ColumnTransformer
) -> Pipeline:
    """Retorna Pipeline(preprocessor, model) para clasificación."""
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate_regression(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Entrena el pipeline y retorna métricas (MAE, RMSE, R2, MAPE) para train y test.

    Returns:
        Dict con claves ``pipeline``, ``train`` y ``test``.
    """
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        }

    return {
        "pipeline": pipeline,
        "train": _metrics(y_train, y_train_pred),
        "test": _metrics(y_test, y_test_pred),
    }


def train_and_evaluate_classification(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Entrena el pipeline y retorna métricas (Accuracy, Precision, Recall, F1, ROC-AUC).

    Requiere que el modelo exponga ``predict_proba``.

    Returns:
        Dict con claves ``pipeline``, ``train`` y ``test``.
    """
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    def _metrics(
        y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> dict[str, float]:
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "F1": float(f1_score(y_true, y_pred, zero_division=0)),
            "ROC_AUC": float(roc_auc_score(y_true, y_proba)),
        }

    return {
        "pipeline": pipeline,
        "train": _metrics(y_train, y_train_pred, y_train_proba),
        "test": _metrics(y_test, y_test_pred, y_test_proba),
    }


def save_model(pipeline: Pipeline, path: str) -> None:
    """Guarda el pipeline en disco con joblib.

    Raises:
        OSError: Si no se puede escribir en ``path``.
    """
    try:
        joblib.dump(pipeline, path)
    except OSError as exc:
        raise OSError(
            f"No se pudo guardar el modelo en '{path}'. "
            f"Verifica permisos y que el directorio exista. Detalle: {exc}"
        ) from exc
