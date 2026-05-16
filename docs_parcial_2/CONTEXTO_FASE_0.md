# Contexto del proyecto — Fase 0 (reorganización de repo)

## Situación actual

- Estamos trabajando en la **Evaluación Parcial 2** del ramo SCY1101 (Programación para la Ciencia de Datos).
- El repo actual (`Nicossm/parcial1`) contiene el trabajo de la **Evaluación Parcial 1**, que fue exclusivamente sobre limpieza y transformación de datos (EDA, manejo de nulos, encoding, escalado).
- El Parcial 2 se construye **encima** de ese trabajo: implementación de modelos de ML, evaluación y optimización.

## Decisión tomada

Reorganizar el repo para que:
1. **Todo el trabajo del Parcial 1 quede preservado** en una carpeta llamada `etl/` (de Extract, Transform, Load — terminología profesional estándar para el pipeline de datos).
2. **La raíz del repo contenga la nueva estructura del Parcial 2** según la especificación oficial de la evaluación.
3. El historial de commits del trabajo previo se preserve usando `git mv`.

## Estructura objetivo

```
salary-ml-project/                 # nombre nuevo del repo (renombrar en GitHub manualmente)
├── etl/                           # TODO el contenido actual del repo va aquí
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── docs/
│   ├── Software_Developer_Salary/
│   └── README.md                  # el README actual se mueve aquí
├── notebooks/                     # NUEVO — Parcial 2
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_supervised_modeling.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_optimization.ipynb
│   └── 05_final_analysis.ipynb
├── src/                           # NUEVO
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── hyperparameter_tuning.py
├── models/                        # NUEVO (modelos serializados con joblib)
├── results/                       # NUEVO (métricas, gráficos, reportes)
└── README.md                      # NUEVO (del Parcial 2)
```

## Problema de negocio definido para el Parcial 2

Sobre el dataset de salarios de desarrolladores de software:
- **Regresión:** predecir el valor exacto de `salary_usd`.
- **Clasificación binaria:** predecir si `salary_usd` supera la mediana del mercado (~130K USD) — variable derivada.

Esto cumple el requisito de IEE 2.1.1 que pide "modelos de clasificación **y** regresión".

## Indicadores que se evaluarán (rúbrica oficial)

### Encargo grupal (10% del Parcial 2)
- IEE 2.1.1: modelos supervisados con pipelines y justificación (20%)
- IEE 2.1.2: aprendizaje no supervisado, clustering y PCA (20%)
- IEE 2.2.1: validación cruzada y múltiples métricas (30%)
- IEE 2.3.1: optimización con GridSearchCV **y** RandomizedSearchCV (30%)

### Presentación individual (20% del Parcial 2)
- IEP 2.1.3: explicar implementación de modelos supervisados (30%)
- IEP 2.2.2: interpretar y comparar métricas (35%)
- IEP 2.3.2: explicar optimización de hiperparámetros (35%)

## Tarea concreta para esta sesión (Fase 0)

1. **Crear la carpeta `etl/`** en la raíz del repo.
2. **Mover con `git mv`** (NO con `mv`, para preservar historial) todo el contenido actual del repo a `etl/`:
   - `notebooks/` → `etl/notebooks/`
   - `src/` → `etl/src/`
   - `data/` → `etl/data/`
   - `docs/` → `etl/docs/`
   - `Software_Developer_Salary/` → `etl/Software_Developer_Salary/`
   - `README.md` → `etl/README.md`
3. **Crear las nuevas carpetas en la raíz:**
   - `notebooks/`, `src/`, `models/`, `results/`
4. **Crear archivos placeholder vacíos:**
   - 5 notebooks en `notebooks/` (con el nombre exacto que pide el PDF).
   - 4 archivos `.py` en `src/` (con el nombre exacto que pide el PDF).
   - Un `README.md` nuevo en la raíz (ver contenido más abajo).
   - Un `.gitkeep` en `models/` y `results/` para que git las registre.
5. **Hacer commit** con un mensaje claro, ejemplo:
   ```
   Reestructura proyecto: trabajo de preparación de datos movido a /etl/, prepara estructura para fase de modelado del Parcial 2
   ```
6. **NO hacer push** — el usuario revisará y hará el push manualmente.
7. **Mostrar `git status` y `git log --oneline -5`** al finalizar para verificar.

## Contenido sugerido para el nuevo README.md principal

El README de la raíz debe enmarcar el proyecto completo (no solo el Parcial 2). Estructura sugerida:

- **Título:** Salary ML Project — Predicción y análisis de salarios de desarrolladores de software
- **Descripción del proyecto** (incluye el problema de negocio: regresión + clasificación binaria sobre mediana del mercado).
- **Estructura del repositorio:**
  - `etl/` — pipeline de extracción, transformación y carga de datos
  - `notebooks/` — análisis y modelado
  - `src/` — módulos Python reutilizables
  - `models/` — modelos entrenados
  - `results/` — métricas y visualizaciones
- **Stack tecnológico:** Python 3.x, scikit-learn, pandas, numpy, matplotlib, seaborn, joblib.
- **Cómo ejecutar:** clonar, instalar dependencias, ejecutar notebooks en orden.
- **Integrantes:** [dejar placeholders para que el usuario los complete después].

El framing debe ser **profesional**, presentando el proyecto como un pipeline completo de ML, NO como dos parciales encadenados. La carpeta `etl/` existe porque cualquier proyecto serio de ML tiene esa fase, no porque sea un entregable previo.

## Restricciones importantes

- **Usar `git mv`, no `mv`.** Preservar historial de cada archivo es crítico.
- **No hacer `git push`.** Solo commit local. El usuario revisa antes de empujar.
- **No tocar el contenido de los archivos del Parcial 1.** Solo moverlos.
- **No crear contenido en los notebooks ni en los `.py` nuevos.** Solo crearlos vacíos como placeholder. El contenido se construye en fases posteriores.
