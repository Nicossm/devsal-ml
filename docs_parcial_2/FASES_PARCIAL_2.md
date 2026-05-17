# Fases de Trabajo — Parcial 2 (SCY1101)

> Proyecto: Salary ML Project
> Entrega: **Martes 19/05/2026 — inicio de clase**
> Defensa individual: **Miércoles 20/05/2026 — en clase**
> Ponderación total: 30% del ramo (10% encargo grupal + 20% presentación individual)

---

## Indicadores de evaluación (lo que el docente mide)

### Encargo grupal (10%)
| Código | Indicador | Peso interno |
|---|---|---|
| IEE 2.1.1 | Modelos supervisados (clasificación + regresión) con Scikit-learn | 20% |
| IEE 2.1.2 | Aprendizaje no supervisado (clustering + reducción de dimensionalidad) | 20% |
| IEE 2.2.1 | Validación cruzada y múltiples métricas con interpretación comparativa | 30% |
| IEE 2.3.1 | Optimización de hiperparámetros (GridSearchCV + RandomizedSearchCV) | 30% |

### Presentación individual (20%)
| Código | Indicador | Peso interno |
|---|---|---|
| IEP 2.1.3 | Explica modelos supervisados y justifica selección | 30% |
| IEP 2.2.2 | Interpreta y compara métricas | 35% |
| IEP 2.3.2 | Explica optimización de hiperparámetros y su impacto | 35% |

---

## Fase 0 — Preparación de la repo

**Duración estimada:** 1 día

**Trabajo:**
- Renombrar repo en GitHub (`parcial1` → `salary-ml-project` o similar).
- Reorganizar estructura: trabajo previo a `etl/`, nueva estructura del Parcial 2 en la raíz.
- Definir dataset a usar en modelado (recomendado: `transformed.csv` que sale de `etl/`).
- Repartir tareas entre el equipo.

**Entregable parcial:** repo reorganizada con la estructura mínima exigida por el PDF.

**Estructura objetivo:**
```
salary-ml-project/
├── etl/                       # Trabajo del Parcial 1
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
├── models/
├── results/
└── README.md
```

---

## Fase 1 — Definir el problema de negocio

**Duración estimada:** 1 día

**Trabajo:**
- Decidir formalmente qué predecir sobre `salary_usd`:
  - **Regresión:** valor exacto del salario.
  - **Clasificación:** binaria (¿supera la mediana?) o multiclase por cuartiles.
- **Recomendado: ambas.** El PDF dice literalmente *"Implementa múltiples modelos de clasificación **y** regresión"* (IEE 2.1.1). Hacer solo una baja la nota en este indicador.

**Entregable parcial:** sección en el README + introducción del notebook 01 justificando el problema.

---

## Fase 2 — EDA condensado

**Duración estimada:** 1 día
**Notebook:** `01_exploratory_analysis.ipynb`

**Trabajo:**
- Carga del dataset.
- Resumen del estado: shape, dtypes, nulos, duplicados.
- Distribución del target (`salary_usd`): histograma, boxplot.
- Distribución de features categóricas y numéricas.
- Correlaciones (heatmap).
- Si se hace clasificación: crear y mostrar el balance de la clase derivada.

**Atajo válido:** como el EDA profundo ya está en `etl/`, este notebook puede ser condensado y enfocado en lo que importa para modelado. Mencionar en el notebook que el preprocesamiento completo está en `etl/`.

**Indicador que cubre:** apoya IEE 2.1.2 (parte de exploración) y prepara el terreno para el resto.

---

## Fase 3 — Modelos supervisados

**Duración estimada:** 2-3 días
**Notebook:** `02_supervised_modeling.ipynb`
**Módulo:** `src/model_training.py`

**Trabajo:**
- Split train/test (`sklearn.model_selection.train_test_split`, `random_state` fijo, `stratify` si es clasificación).
- Encapsular preprocesamiento + modelo en `sklearn.pipeline.Pipeline`.

**Regresión — implementar al menos 3-4 modelos:**
- Linear Regression / Ridge / Lasso
- Random Forest Regressor
- Gradient Boosting Regressor
- SVR o KNN Regressor

**Clasificación — implementar al menos 3-4 modelos:**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- SVM o KNN Classifier

**Tareas adicionales:**
- Entrenar cada modelo, guardar resultados base (sin tuning aún).
- Serializar modelos en `models/` con `joblib.dump`.
- Justificar técnicamente cada elección de modelo en celdas markdown.

**Indicador que cubre:** **IEE 2.1.1 (20%)**.

**Para alcanzar 100%:** *"Implementa y configura múltiples modelos con pipelines, justifica técnicamente cada decisión y demuestra dominio de Scikit-learn"*. Los pipelines y la justificación escrita separan el 80% del 100%.

---

## Fase 4 — Aprendizaje no supervisado

**Duración estimada:** 1-2 días
**Notebook:** sección dedicada (puede ir en `02_supervised_modeling.ipynb` o en `05_final_analysis.ipynb`)

**Trabajo:**
- **Clustering:**
  - K-Means con método del codo + silhouette score para elegir K.
  - Opcional: DBSCAN o clustering jerárquico para comparar.
- **Reducción de dimensionalidad:**
  - PCA para visualización 2D de los clusters.
  - Mostrar varianza explicada por componente.
- **Interpretación:** ¿qué tipo de perfil de desarrollador representa cada cluster? (ej. "senior backend en empresa grande", "junior frontend en startup").
- **Visualizaciones:** scatter de PCA coloreado por cluster, distribuciones por cluster.

**Indicador que cubre:** **IEE 2.1.2 (20%)**.

**Para alcanzar 100%:** *"Aplica múltiples técnicas no supervisadas, selecciona y evalúa con métricas y visualizaciones avanzadas"*. Justificar el K elegido con elbow + silhouette es lo que sube de 80% a 100%.

---

## Fase 5 — Evaluación y comparación

**Duración estimada:** 1-2 días
**Notebook:** `03_model_evaluation.ipynb`
**Módulo:** `src/model_evaluation.py`

**Trabajo:**
- **Validación cruzada** con `cross_val_score` (cv=5 o 10) para cada modelo.
- **Múltiples métricas:**
  - Regresión: MAE, RMSE, R², MAPE.
  - Clasificación: Accuracy, Precision, Recall, F1, ROC-AUC.
- **Matriz de confusión** para clasificación.
- **Tabla comparativa** de todos los modelos (DataFrame exportable a `results/`).
- **Visualizaciones:** bar plots comparativos, curvas ROC, gráficos de residuos para regresión.
- Análisis de overfitting/underfitting comparando train vs validation scores.

**Indicador que cubre:** **IEE 2.2.1 (30%)** — el indicador más pesado del encargo junto con el de optimización.

**Para alcanzar 100%:** *"Realiza validación cruzada robusta, calcula todas las métricas clave, compara y visualiza resultados con análisis avanzado"*. "Robusta" y "avanzado" implican interpretar y visualizar, no solo correr funciones y mostrar números.

---

## Fase 6 — Optimización de hiperparámetros

**Duración estimada:** 2 días
**Notebook:** `04_hyperparameter_optimization.ipynb`
**Módulo:** `src/hyperparameter_tuning.py`

**Trabajo:**
- Seleccionar los 2-3 modelos con mejor desempeño base.
- **GridSearchCV** sobre uno (típicamente Random Forest o Gradient Boosting).
- **RandomizedSearchCV** sobre otro.
- **Justificar el grid:** por qué esos valores, por qué ese rango.
- Comparar rendimiento antes y después del tuning.
- Visualizar el impacto: bar chart pre/post, o heatmap de resultados del grid.
- Documentar los mejores hiperparámetros encontrados.

**Indicador que cubre:** **IEE 2.3.1 (30%)**.

**Para alcanzar 100%:** *"Implementa optimización exhaustiva y justifica técnica y visualmente el proceso y sus efectos"*. Usar **ambos** métodos (Grid y Randomized) es clave — usar solo uno baja a "buen desempeño" (80%).

---

## Fase 7 — Análisis final e integración

**Duración estimada:** 1 día
**Notebook:** `05_final_analysis.ipynb`

**Trabajo:**
- Recomendación del mejor modelo con justificación técnica.
- Reflexión sobre trade-offs (interpretabilidad vs rendimiento, velocidad vs precisión).
- Feature importance del modelo ganador.
- Limitaciones del proyecto.
- Conexión con el análisis no supervisado: ¿los clusters ayudan a entender errores del modelo?
- **Opcional (si hay tiempo):** validar contra `train.csv` + `test.csv` (50K filas adicionales) como hold-out externo.

---

## Fase 8 — Entregables finales

**Duración estimada:** 1-2 días (en paralelo con fases anteriores)

### Informe técnico (12-15 páginas)
- Resumen ejecutivo
- Marco metodológico (justificación de algoritmos y técnicas)
- Análisis experimental (descripción de experimentos, datos, configuraciones)
- Resultados y comparación de modelos (métricas, tablas, gráficos)
- Optimización de hiperparámetros (proceso y análisis de impacto)
- Conclusiones y recomendaciones
- Referencias bibliográficas

### Video grupal (5-7 minutos)
- **TODOS los integrantes deben hablar.**
- Mostrar los notebooks ejecutándose realmente (no solo slides).
- Cubrir los 8 puntos exigidos por el PDF:
  1. Análisis exploratorio de datos
  2. Limpieza y preparación
  3. Modelos supervisados
  4. Técnicas no supervisadas
  5. Comparación de métricas
  6. GridSearchCV / RandomizedSearchCV
  7. Resultados obtenidos
  8. Participación técnica de todos los integrantes

### README.md del repo
- Descripción del proyecto y problema de negocio
- Estructura del repositorio
- Instrucciones de uso (clonar, dependencias, ejecutar)
- Resumen de resultados
- Integrantes del equipo

### Subida a AVA
- Link del repositorio GitHub
- Link del video
- Informe técnico (PDF)
- **Cada integrante registra individualmente la entrega.**

---

## Cronograma sugerido

| Día | Foco |
|---|---|
| Sábado 16/05 | Fases 0, 1, 2 (preparación, problema, EDA condensado) |
| Domingo 17/05 | Fase 3 (modelos supervisados — el bloque más largo) |
| Lunes 18/05 | Fases 4 y 5 (no supervisado + evaluación con CV) |
| Martes 19/05 AM | Fase 6 (optimización) + Fase 7 (análisis final) |
| Martes 19/05 antes de la clase | Video, informe, push final, subir a AVA |

---

## Recordatorios críticos

1. **Pipelines de sklearn:** encapsular preprocesamiento + modelo en un solo `Pipeline` no es opcional para alcanzar el 100% en IEE 2.1.1.

2. **Tanto Grid como Randomized:** la rúbrica IEE 2.3.1 exige ambos métodos. Solo uno = máximo 80%.

3. **Justificación escrita:** cada decisión técnica (elección de modelo, valor de K, rango de hiperparámetros) debe ir acompañada de una celda markdown justificándola. La defensa individual del miércoles pregunta exactamente por estas justificaciones.

4. **`random_state` fijo en todo:** train_test_split, modelos, KFold. El proyecto debe ser 100% reproducible.

5. **Docstrings en `.py`:** las funciones en `src/` deben tener docstrings. Es parte del aspecto formal del código.

6. **La defensa individual del miércoles es sin internet.** Cada integrante debe entender **todo** el código, no solo su parte.

7. **Subir antes del inicio de la clase del martes.** El docente ocupa esa clase para revisar videos y preparar las preguntas individuales.
