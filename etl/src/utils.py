"""
Módulo de utilidades generales.
Contiene funciones auxiliares para carga, guardado y análisis de datos.
"""

import pandas as pd
import numpy as np
import os


def load_data(filepath, **kwargs):
    """
    Carga un archivo de datos en un DataFrame.
    
    Soporta formatos: CSV, Excel, JSON, Parquet.
    
    Args:
        filepath (str): Ruta al archivo de datos
        **kwargs: Argumentos adicionales para el lector
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    
    Raises:
        ValueError: Si el formato de archivo no es soportado
    
    Example:
        >>> df = load_data('data/raw/dataset.csv')
        >>> df = load_data('data/raw/dataset.xlsx', sheet_name='Sheet1')
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    readers = {
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel,
        '.json': pd.read_json,
        '.parquet': pd.read_parquet
    }
    
    if ext not in readers:
        raise ValueError(f"Formato no soportado: {ext}. Formatos válidos: {list(readers.keys())}")
    
    df = readers[ext](filepath, **kwargs)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def save_data(df, filepath, index=False, **kwargs):
    """
    Guarda un DataFrame en un archivo.
    
    Soporta formatos: CSV, Excel, JSON, Parquet.
    
    Args:
        df (pd.DataFrame): DataFrame a guardar
        filepath (str): Ruta donde guardar el archivo
        index (bool): Si incluir el índice en el archivo
        **kwargs: Argumentos adicionales para el escritor
    
    Example:
        >>> save_data(df, 'data/processed/dataset_clean.csv')
    """
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if ext == '.csv':
        df.to_csv(filepath, index=index, **kwargs)
    elif ext in ['.xlsx', '.xls']:
        df.to_excel(filepath, index=index, **kwargs)
    elif ext == '.json':
        df.to_json(filepath, **kwargs)
    elif ext == '.parquet':
        df.to_parquet(filepath, index=index, **kwargs)
    else:
        raise ValueError(f"Formato no soportado: {ext}")
    
    print(f"Datos guardados en: {filepath}")


def get_data_quality_report(df):
    """
    Genera un reporte de calidad de datos del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    
    Returns:
        pd.DataFrame: Reporte con métricas de calidad por columna
    
    Example:
        >>> report = get_data_quality_report(df)
        >>> print(report)
    """
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'non_null': df.count(),
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'unique_pct': (df.nunique() / len(df) * 100).round(2)
    })
    
    # Agregar estadísticas para columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        report.loc[col, 'min'] = df[col].min()
        report.loc[col, 'max'] = df[col].max()
        report.loc[col, 'mean'] = df[col].mean()
        report.loc[col, 'std'] = df[col].std()
    
    return report


def get_duplicate_report(df, subset=None):
    """
    Genera un reporte de filas duplicadas.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        subset (list, optional): Columnas a considerar para identificar duplicados
    
    Returns:
        dict: Diccionario con información de duplicados
    
    Example:
        >>> dup_report = get_duplicate_report(df)
        >>> print(f"Duplicados: {dup_report['count']}")
    """
    duplicates = df.duplicated(subset=subset, keep=False)
    
    return {
        'count': duplicates.sum(),
        'percentage': (duplicates.sum() / len(df) * 100).round(2),
        'unique_duplicated': df[duplicates].drop_duplicates().shape[0] if duplicates.any() else 0,
        'columns_checked': subset if subset else 'all'
    }


def detect_outliers_iqr(df, column, threshold=1.5):
    """
    Detecta outliers usando el método IQR.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        column (str): Nombre de la columna numérica
        threshold (float): Multiplicador del IQR (default 1.5)
    
    Returns:
        dict: Diccionario con información de outliers
    
    Example:
        >>> outliers = detect_outliers_iqr(df, 'salary_usd')
        >>> print(f"Outliers encontrados: {outliers['count']}")
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    return {
        'column': column,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'count': outliers_mask.sum(),
        'percentage': (outliers_mask.sum() / len(df) * 100).round(2),
        'outlier_values': df.loc[outliers_mask, column].tolist()[:10]  # Primeros 10
    }


def compare_datasets(df_before, df_after, name_before='Original', name_after='Procesado'):
    """
    Compara dos DataFrames y muestra las diferencias.
    
    Args:
        df_before (pd.DataFrame): DataFrame original
        df_after (pd.DataFrame): DataFrame procesado
        name_before (str): Nombre para el DataFrame original
        name_after (str): Nombre para el DataFrame procesado
    
    Returns:
        pd.DataFrame: Tabla comparativa
    
    Example:
        >>> comparison = compare_datasets(df_raw, df_clean)
        >>> print(comparison)
    """
    comparison = pd.DataFrame({
        name_before: {
            'Filas': len(df_before),
            'Columnas': len(df_before.columns),
            'Valores Nulos Totales': df_before.isnull().sum().sum(),
            'Duplicados': df_before.duplicated().sum(),
            'Memoria (MB)': df_before.memory_usage(deep=True).sum() / 1024**2
        },
        name_after: {
            'Filas': len(df_after),
            'Columnas': len(df_after.columns),
            'Valores Nulos Totales': df_after.isnull().sum().sum(),
            'Duplicados': df_after.duplicated().sum(),
            'Memoria (MB)': df_after.memory_usage(deep=True).sum() / 1024**2
        }
    })
    
    comparison['Diferencia'] = comparison[name_after] - comparison[name_before]
    comparison['Cambio (%)'] = ((comparison[name_after] - comparison[name_before]) / comparison[name_before] * 100).round(2)
    
    return comparison


def print_section(title, char='=', width=60):
    """
    Imprime un título de sección formateado.
    
    Args:
        title (str): Título de la sección
        char (str): Caracter para la línea decorativa
        width (int): Ancho total de la línea
    
    Example:
        >>> print_section("Análisis Exploratorio")
        ============================================================
        ANÁLISIS EXPLORATORIO
        ============================================================
    """
    print(char * width)
    print(title.upper().center(width))
    print(char * width)
