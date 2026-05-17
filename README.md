# Salary ML Project

Prediccion y analisis de salarios de desarrolladores de software.

## Descripcion del proyecto

Pipeline completo de Machine Learning para predecir y analizar salarios de desarrolladores de software, abarcando desde la preparacion de datos hasta la optimizacion de modelos.

### Problema de negocio

- **Regresion:** predecir el valor exacto del salario anual en USD (`salary_usd`).
- **Clasificacion binaria:** predecir si un desarrollador supera la mediana del mercado (~130K USD), variable derivada a partir de `salary_usd`.

## Estructura del repositorio

```
salary-ml-project/
├── etl/                   # Pipeline de extraccion, transformacion y carga de datos
├── notebooks/             # Analisis exploratorio y modelado
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/                   # Modulos Python reutilizables
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
├── models/                # Modelos entrenados serializados (joblib)
├── results/               # Metricas, graficos y reportes
└── README.md
```

## Stack tecnologico

- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- joblib

## Como ejecutar

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Nicossm/parcial1.git
   cd parcial1
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar los notebooks en orden numerico (`01_` a `05_`).

## Integrantes

- [Nombre 1]
- [Nombre 2]
- [Nombre 3]
