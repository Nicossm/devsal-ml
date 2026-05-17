"""Modulo de entrenamiento de modelos supervisados.

Funciones reutilizables para construir pipelines de scikit-learn,
entrenar modelos de regresion y clasificacion, calcular metricas y
serializar los modelos resultantes.

Este modulo es usado por el notebook ``02_supervised_modeling.ipynb``.
La logica de entrenamiento vive aqui para mantener el notebook como
una capa de orquestacion y narracion, no de implementacion.
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
    """Carga el dataset limpio desde disco y valida su esquema.

    Args:
        path: Ruta al archivo CSV (tipicamente ``etl/data/processed/clean.csv``).

    Returns:
        DataFrame con las columnas esperadas del dataset limpio.

    Raises:
        FileNotFoundError: Si el archivo no existe en ``path``.
        ValueError: Si faltan columnas requeridas en el archivo.
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
    """Construye la variable binaria ``salary_above_median`` sin leakage.

    El umbral se calcula exclusivamente sobre ``y_train`` (mediana de train)
    y luego se aplica tanto a train como a test. Asi se evita filtrar
    informacion del conjunto de test al modelo.

    Args:
        y_train: Salarios numericos del conjunto de entrenamiento.
        y_test: Salarios numericos del conjunto de prueba.

    Returns:
        Tupla ``(y_train_binary, y_test_binary, threshold)``:
        - ``y_train_binary``: 1 si el salario supera la mediana de train.
        - ``y_test_binary``: 1 si el salario supera la mediana de train.
        - ``threshold``: valor usado como umbral (mediana de train).

    Raises:
        TypeError: Si ``y_train`` o ``y_test`` no son ``pd.Series``.
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
    """Construye un ColumnTransformer con escalado y one-hot encoding.

    - Variables numericas: ``StandardScaler`` (justificacion: SVM, Logistic
      Regression y Ridge son sensibles a la escala).
    - Variables categoricas: ``OneHotEncoder`` con ``handle_unknown='ignore'``
      (justificacion: robustez frente a categorias no vistas en validacion
      cruzada y test).

    Args:
        numeric_features: Nombres de las columnas numericas.
        categorical_features: Nombres de las columnas categoricas.

    Returns:
        ColumnTransformer listo para integrarse en un ``Pipeline``.

    Raises:
        ValueError: Si alguna lista de features esta vacia.
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
    """Encapsula preprocessor + modelo de regresion en un ``Pipeline``.

    Args:
        model: Estimador de regresion compatible con sklearn (ej. ``Ridge``).
        preprocessor: ``ColumnTransformer`` construido con ``build_preprocessor``.

    Returns:
        Pipeline con pasos ``preprocessor`` y ``model``.
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def build_classification_pipeline(
    model: Any, preprocessor: ColumnTransformer
) -> Pipeline:
    """Encapsula preprocessor + modelo de clasificacion en un ``Pipeline``.

    Args:
        model: Estimador de clasificacion compatible con sklearn.
        preprocessor: ``ColumnTransformer`` construido con ``build_preprocessor``.

    Returns:
        Pipeline con pasos ``preprocessor`` y ``model``.
    """
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_and_evaluate_regression(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Entrena un pipeline de regresion y calcula metricas en train y test.

    Metricas calculadas: MAE, RMSE, R^2 y MAPE.

    Args:
        pipeline: Pipeline con preprocesamiento + modelo de regresion.
        X_train: Features de entrenamiento.
        y_train: Target numerico de entrenamiento.
        X_test: Features de test.
        y_test: Target numerico de test.

    Returns:
        Dict con claves ``pipeline``, ``train`` y ``test``. Cada uno de
        ``train`` y ``test`` contiene MAE, RMSE, R2 y MAPE.
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
    """Entrena un pipeline de clasificacion binaria y calcula metricas.

    Metricas calculadas: Accuracy, Precision, Recall, F1 y ROC-AUC.
    ROC-AUC requiere que el modelo exponga ``predict_proba``; los modelos
    incluidos en este proyecto (Logistic, SVM con ``probability=True``,
    Random Forest, Gradient Boosting) cumplen este requisito.

    Args:
        pipeline: Pipeline con preprocesamiento + clasificador.
        X_train: Features de entrenamiento.
        y_train: Target binario de entrenamiento.
        X_test: Features de test.
        y_test: Target binario de test.

    Returns:
        Dict con claves ``pipeline``, ``train`` y ``test``. Cada uno de
        ``train`` y ``test`` contiene Accuracy, Precision, Recall, F1 y
        ROC-AUC.
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
    """Serializa un pipeline entrenado a disco con ``joblib.dump``.

    Args:
        pipeline: Pipeline entrenado a persistir.
        path: Ruta destino (extension ``.joblib`` recomendada).

    Raises:
        OSError: Si no se puede escribir en la ruta indicada.
    """
    try:
        joblib.dump(pipeline, path)
    except OSError as exc:
        raise OSError(
            f"No se pudo guardar el modelo en '{path}'. "
            f"Verifica permisos y que el directorio exista. Detalle: {exc}"
        ) from exc
