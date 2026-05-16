# Proyecto Parcial 1: Análisis de Salarios de Desarrolladores de Software

## Integrantes

| Nombre | Rol |
|--------|-----|
| Nicolás Osses | Integrante 1 — Setup y Limpieza de Datos |
| Rolando Paredes | Integrante 2 — Transformación y Pipeline |
| Belén Toloza | Integrante 3 — Feature Engineering, Visualización y README |

## Descripción del Proyecto

Este proyecto forma parte de la Evaluación Parcial N°1 de la asignatura **Programación para la Ciencia de Datos (SCY1101)** del **Instituto Profesional Duoc UC**.

El objetivo es aplicar técnicas de manipulación, limpieza y transformación de datos sobre un dataset real de salarios de desarrolladores de software, preparando la información para etapas futuras de modelado con Machine Learning.

## Dataset

### Origen
- **Fuente**: Kaggle — Software Developer Salary Dataset
- **Tema**: Salarios de desarrolladores de software a nivel mundial
- **Variable objetivo**: `salary_usd` (salario anual en USD)

### Descripción de las Variables

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `experience` | Numérico | Años de experiencia profesional |
| `country` | Categórico | País de residencia |
| `education` | Categórico | Nivel educativo más alto |
| `languages` | Texto | Lenguajes de programación que maneja |
| `frameworks` | Texto | Frameworks y tecnologías que utiliza |
| `company_size` | Categórico | Tamaño de la empresa |
| `salary_usd` | Numérico | Salario anual en USD (variable objetivo) |

### Estado del Dataset Raw

El dataset crudo presentaba las siguientes impurezas que fueron tratadas a lo largo del proyecto:

- **Valores nulos**: presentes en varias columnas, incluyendo pseudonulos como `"missing"`, `"N/A"`, `"unknown"`
- **Duplicados**: filas completamente repetidas
- **Outliers**: valores atípicos en `salary_usd` y `experience`
- **Inconsistencias de formato**: variaciones en nombres de países (ej. `"USA"`, `"US"`, `"United States"`) y niveles educativos
- **Tipos mezclados**: valores de texto en columnas numéricas

## Estructura del Proyecto

```
parcial1/
├── data/
│   ├── raw/
│   │   └── software_developer_salary_raw.csv
│   ├── processed/
│   │   └── clean.csv
│   ├── transformed/
│   │   └── transformed.csv
│   └── featured/
│       └── featured.csv
├── docs/
│   └── Evaluación Parcial 1.pdf
├── notebooks/
│   ├── 01_limpieza.ipynb
│   ├── 02_transformacion.ipynb
│   └── 03_feature_engineering_visualizacion.ipynb
├── src/
└── README.md
```

## Entorno de Desarrollo

### Google Colab

El equipo trabajó en **Google Colab** por las siguientes razones:

- No requiere instalación local de Python ni librerías, cualquier integrante puede abrir el notebook desde el navegador
- Permite cargar archivos directamente desde URLs raw de GitHub, lo que facilita la reproducibilidad
- Al correr en el mismo entorno cloud, los tres integrantes obtenemos los mismos resultados sin diferencias por versiones o sistemas operativos distintos

### Librerías utilizadas

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Todas disponibles por defecto en Google Colab, sin necesidad de instalar nada.

## Cómo Ejecutar

1. Abrir Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Ir a `File → Open notebook → GitHub`
3. Pegar la URL del repositorio: `https://github.com/Nicossm/parcial1`
4. Ejecutar los notebooks **en orden**:
   - `01_limpieza.ipynb`
   - `02_transformacion.ipynb`
   - `03_feature_engineering_visualizacion.ipynb`

> Cada notebook carga los datos directamente desde GitHub mediante URL, por lo que no es necesario descargar ningún archivo manualmente.

## Metodología

### 1. Limpieza de Datos — Nicolás Osses

- Unificación de pseudonulos a formato `NaN`
- Normalización de variables categóricas (`country`, `education`)
- Eliminación de duplicados
- Tratamiento de valores imposibles en `experience` y `salary_usd`
- Imputación de `experience` con mediana
- Tratamiento de outliers en `salary_usd` mediante capping con IQR

### 2. Transformación y Pipeline — Rolando Paredes

- Escalado de variables numéricas con `StandardScaler`
- Codificación de variables categóricas con `OneHotEncoder`
- Construcción de un `Pipeline` con `ColumnTransformer` de scikit-learn
- Justificación técnica de cada decisión de transformación

### 3. Feature Engineering y Visualización — Belén Toloza

- Creación de nuevas variables derivadas:
  - `is_high_education`: indica si el desarrollador tiene Masters o PhD
  - `is_big_company`: indica si trabaja en empresa de más de 1000 empleados
  - `exp_edu_score`: score combinado de experiencia y nivel educativo
- Visualizaciones exploratorias con Matplotlib y Seaborn
- Redacción del README

## Conclusiones

El proceso de limpieza y transformación permitió convertir un dataset crudo con múltiples impurezas en un dataset completamente numérico, sin nulos y listo para ser usado en algoritmos de Machine Learning. Las nuevas variables creadas en la fase de Feature Engineering, especialmente `exp_edu_score`, mostraron mayor correlación con el salario que las variables originales por separado, lo que aporta valor al futuro modelo predictivo.

---

*Proyecto académico — Instituto Profesional Duoc UC — SCY1101 Programación para la Ciencia de Datos*
