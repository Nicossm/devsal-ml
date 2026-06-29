# Developer Salary Prediction

Pipeline end-to-end de Ciencia de Datos sobre el dataset **Software Developer Salary**, abarcando limpieza, transformación, modelado supervisado y no supervisado, evaluación y optimización de hiperparámetros.

Proyecto académico de la asignatura **SCY1101 — Programación para la Ciencia de Datos** del Instituto Profesional Duoc UC. Cubre las dos evaluaciones del ramo:

- **Parcial 1** — ETL y feature engineering (carpeta `etl/`)
- **Parcial 2** — Modelado, evaluación, tuning y clustering (raíz del repo)

## Problema de negocio

A partir del perfil de un desarrollador (experiencia, país, educación, stack, tamaño de empresa) se abordan tres tareas:

| Tarea | Tipo | Variable objetivo |
|---|---|---|
| Predicción de salario | Regresión | `salary_usd` |
| Segmentación por mediana de mercado (~130K USD) | Clasificación binaria | derivada de `salary_usd` |
| Perfiles de desarrolladores | Clustering (no supervisado) | — |

## Dataset

- **Fuente:** Kaggle — *Software Developer Salary*
- **Tamaño raw:** 10.300 filas · 7 columnas
- **Tamaño limpio:** 9.249 filas tras tratamiento de nulos, duplicados y outliers

| Columna | Tipo | Descripción |
|---|---|---|
| `experience` | Numérico | Años de experiencia (0–40) |
| `country` | Categórico | País de residencia (10 países) |
| `education` | Categórico | Nivel educativo (5 niveles) |
| `languages` | Texto multivalor | Lenguajes de programación |
| `frameworks` | Texto multivalor | Frameworks y tecnologías |
| `company_size` | Categórico | Tamaño de empresa (6 rangos) |
| `salary_usd` | Numérico | Salario anual en USD — **target** |

## Estructura del repositorio

```
.
├── etl/                      # Parcial 1 — ETL y feature engineering
│   ├── data/
│   │   ├── raw/              # Dataset original
│   │   ├── processed/        # clean.csv
│   │   ├── transformed/      # transformed.csv (encoded + scaled)
│   │   └── featured/         # featured.csv
│   ├── notebooks/            # 01_limpieza, 02_transformacion, 03_feature_eng
│   ├── src/                  # Módulos de limpieza y transformación
│   └── README.md
│
├── notebooks/                # Parcial 2 — Modelado
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   ├── 05_final_analysis.ipynb
│   ├── 06_unsupervised_modeling.ipynb
│   ├── 06b_clustering_evaluation.ipynb
│   ├── 06c_clustering_tuning.ipynb
│   └── 06d_clustering_optimization.ipynb
│
├── src/                      # Módulos reutilizables (Parcial 2)
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
│
├── models/                   # Modelos entrenados serializados (.joblib)
├── results/                  # Métricas (CSV) y gráficos (PNG)
└── docs_parcial_2/           # Enunciados y contexto de la evaluación
```

## Modelos implementados

**Regresión** — Linear, Ridge, Random Forest, Gradient Boosting
**Clasificación** — Logistic Regression, Random Forest, SVM, Gradient Boosting
**Clustering** — KMeans con reducción de dimensionalidad vía PCA

Cada familia se evalúa con validación cruzada, se optimiza con búsqueda de hiperparámetros y se serializa en `models/` (versiones base y `tuned_*`).

## Resultados

La carpeta `results/` contiene los artefactos generados por los notebooks:

- `cv_regression.csv` / `cv_classification.csv` — métricas de validación cruzada
- `tuning_regression.csv` / `tuning_classification.csv` — comparativa pre/post tuning
- `confusion_matrices.png`, `roc_curves.png` — diagnósticos de clasificación
- `residuals.png`, `best_regressor_diagnostics.png` — diagnósticos de regresión
- `clusters.csv` — asignación de clusters por perfil

## Entorno de ejecución

Los notebooks están preparados para correr en **Google Colab**. Los datasets se cargan directamente desde el repositorio vía URL raw de GitHub, por lo que no requiere descargas manuales ni instalación local.

### Stack

```
Python 3.x
pandas · numpy
scikit-learn
matplotlib · seaborn
joblib
```

Todas las dependencias vienen preinstaladas en Colab.

### Cómo ejecutar

1. Abrir [colab.research.google.com](https://colab.research.google.com)
2. `File → Open notebook → GitHub` y pegar la URL del repositorio
3. Ejecutar los notebooks en orden numérico:
   - **Parcial 1:** `etl/notebooks/01_` → `02_` → `03_`
   - **Parcial 2:** `notebooks/01_` → `02_` → ... → `06d_`

## Equipo

| Integrante | Responsabilidad |
|---|---|
| Nicolás Osses | Setup, limpieza de datos y modelado |
| Rolando Paredes | Transformación y pipeline |



## Video Presentación 

https://drive.google.com/drive/folders/1x8dd1PyCLKWk1BO3jwZ9LuPju0z1jVsQ?usp=sharing

---

*Instituto Profesional Duoc UC — SCY1101 Programación para la Ciencia de Datos*
