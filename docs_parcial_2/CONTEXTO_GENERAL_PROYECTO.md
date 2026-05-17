# Contexto General del Proyecto — Resumen de Conversación

## Proyecto
**Evaluación Parcial 2** del ramo **SCY1101 — Programación para la Ciencia de Datos** (Duoc UC).

## Fechas críticas
- **Entrega de artefactos:** Martes 19/05/2026 — antes del inicio de clase
- **Defensa técnica individual:** Miércoles 20/05/2026 — en horario de clase, sin internet
- **Ponderación:** 30% del ramo (10% encargo grupal + 20% presentación individual)

## Repo de GitHub
- **URL:** https://github.com/Nicossm/developer-salary-prediction (renombrado desde `parcial1`)
- **Rama principal de trabajo:** `nicolas` (se mergea a `main` cuando esté listo)

## Dataset
**Software Developer Salary** — salarios de desarrolladores de software a nivel mundial.

Archivos disponibles:
- `software_developer_salary_raw.csv` (10,300 filas, sucio — nulos, duplicados, outliers, valores basura)
- `clean.csv` (9,249 filas — limpio, con feature engineering: `num_languages`, `num_frameworks`, capping de outliers)
- `transformed.csv` (9,249 filas — clean + one-hot encoding + StandardScaler)
- `train.csv` (40,000 filas — limpio, origen diferente al raw, posible validación externa)
- `test.csv` (10,000 filas — limpio, disjunto de train)

**Variables (7 columnas):**
| Columna | Tipo | Descripción |
|---|---|---|
| experience | Numérico | Años de experiencia (0-40) |
| country | Categórico | País de residencia (10 países, USA dominante con ~4000 filas) |
| education | Categórico | Nivel educativo (5 niveles) |
| languages | Texto multi-valor | Lenguajes de programación (10 individuales, combinados de a 2, separados por coma) |
| frameworks | Texto multi-valor | Frameworks (10 individuales, combinados de a 2, separados por coma) |
| company_size | Categórico | Tamaño de empresa (6 rangos) |
| salary_usd | Numérico | Salario anual en USD (12K-281K, media ~130K) — **TARGET** |

**Hallazgos clave del análisis:**
- Correlación experience-salary: 0.77 (señal fuerte)
- Hay filas con valores duplicados en languages/frameworks (ej. "Java, Java")
- Country desbalanceado (USA = 3999, Singapore = 213)
- Datos probablemente sintéticos

## Problema de negocio definido
**"Predicción y segmentación de salarios en el mercado tech global"** con dos tareas:
- **Regresión:** predecir `salary_usd` exacto
- **Clasificación binaria:** ¿el perfil supera la mediana del mercado? (~130K USD, variable derivada de `salary_usd`)
- **No supervisado:** clustering de perfiles de desarrolladores + PCA para visualización

## Decisiones tomadas

### Sobre el preprocesamiento
- El preprocesamiento completo ya fue hecho en el **Parcial 1** y está preservado en la carpeta `etl/` del repo.
- Para el Parcial 2 se trabaja partiendo de los datos ya limpios — no se repite el preprocesamiento completo.
- En el `01_exploratory_analysis.ipynb` se hace un EDA **condensado**, enfocado en lo relevante para modelado, mencionando que el trabajo de limpieza detallado está en `etl/`.
- El archivo `src/data_preprocessing.py` del Parcial 2 es una refactorización/resumen del código del Parcial 1, con funciones limpias y docstrings.

### Sobre la estructura del repo
- Todo el trabajo del Parcial 1 fue movido a `etl/` (Extract, Transform, Load) usando `git mv` para preservar historial.
- La raíz del repo contiene la estructura nueva del Parcial 2.
- Se usó el nombre `etl/` en vez de `parcial1/` por ser terminología profesional estándar.
- **Fase 0 ya completada** — la reorganización del repo fue ejecutada con Claude Code.

### Estructura actual del repo
```
developer-salary-prediction/
├── etl/                           # Trabajo del Parcial 1 (preservado intacto)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── docs/
│   └── README.md
├── notebooks/                     # Parcial 2 (nuevo)
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/                           # Módulos Python (nuevo)
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
├── models/                        # Modelos serializados (nuevo)
├── results/                       # Métricas, gráficos, reportes (nuevo)
└── README.md                      # README del Parcial 2 (nuevo)
```

## Indicadores de evaluación (rúbrica oficial del PDF)

### Encargo grupal (10% del Parcial 2)
| Código | Indicador | Peso |
|---|---|---|
| IEE 2.1.1 | Modelos supervisados (clasificación + regresión) con pipelines Scikit-learn y justificación | 20% |
| IEE 2.1.2 | No supervisado: clustering + reducción de dimensionalidad (PCA) | 20% |
| IEE 2.2.1 | Validación cruzada robusta + múltiples métricas + interpretación comparativa | 30% |
| IEE 2.3.1 | Optimización con GridSearchCV **Y** RandomizedSearchCV, documentando impacto | 30% |

### Presentación individual (20% del Parcial 2)
| Código | Indicador | Peso |
|---|---|---|
| IEP 2.1.3 | Explica modelos supervisados y justifica selección de algoritmos | 30% |
| IEP 2.2.2 | Interpreta y compara métricas en el contexto del problema | 35% |
| IEP 2.3.2 | Explica optimización de hiperparámetros y su impacto | 35% |

## Fases de trabajo (plan completo)

| Fase | Descripción | Estado |
|---|---|---|
| **Fase 0** | Reorganización del repo (etl/ + estructura nueva) | ✅ Completada |
| **Fase 1** | Definir problema de negocio (regresión + clasificación) | ⏳ Pendiente |
| **Fase 2** | EDA condensado (`01_exploratory_analysis.ipynb`) | ⏳ Pendiente |
| **Fase 3** | Modelos supervisados (`02_supervised_modeling.ipynb` + `src/model_training.py`) | ⏳ Pendiente |
| **Fase 4** | No supervisado: K-Means, PCA, silhouette, elbow | ⏳ Pendiente |
| **Fase 5** | Evaluación: CV, métricas, comparación, visualización (`03_model_evaluation.ipynb`) | ⏳ Pendiente |
| **Fase 6** | Optimización: GridSearchCV + RandomizedSearchCV (`04_hyperparameter_optimization.ipynb`) | ⏳ Pendiente |
| **Fase 7** | Análisis final e integración (`05_final_analysis.ipynb`) | ⏳ Pendiente |
| **Fase 8** | Entregables: informe técnico (12-15 pags), video (5-7 min), README, subir a AVA | ⏳ Pendiente |

## Criterios clave para alcanzar 100% en cada indicador
- **IEE 2.1.1:** pipelines de sklearn (no modelos sueltos), justificación técnica escrita en markdown de cada modelo elegido
- **IEE 2.1.2:** múltiples técnicas (K-Means + otra), justificar K con elbow + silhouette, visualizaciones avanzadas con PCA
- **IEE 2.2.1:** CV robusta (cv=5 o 10), TODAS las métricas (MAE, RMSE, R², MAPE, Accuracy, Precision, Recall, F1, ROC-AUC), visualización comparativa, análisis overfitting/underfitting
- **IEE 2.3.1:** usar AMBOS métodos (Grid + Randomized), justificar el grid de parámetros, mostrar antes/después con visualizaciones

## Recordatorios importantes
- `random_state` fijo en todo (reproducibilidad)
- Docstrings en todas las funciones de `src/`
- Código limpio, modular, documentado
- La defensa individual del miércoles es **sin internet** — todos deben entender todo el código
- Video debe mostrar notebooks ejecutándose realmente, no solo slides
- Cada integrante debe registrar individualmente la entrega en AVA

## Archivos de referencia generados
- `FASES_PARCIAL_2.md` — plan detallado de las 9 fases con duración, indicadores, criterios de éxito
- `CONTEXTO_FASE_0.md` — instrucciones puntuales para la reorganización del repo (ya ejecutado)

## Siguiente paso
**Fase 1 + Fase 2:** definir formalmente el problema de negocio y construir el `01_exploratory_analysis.ipynb` condensado.
