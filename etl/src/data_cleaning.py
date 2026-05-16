"""
Módulo de limpieza de datos.
Contiene funciones para manejar valores nulos, duplicados, outliers e inconsistencias.
"""

import pandas as pd
import numpy as np


def remove_duplicates(df, subset=None, keep='first'):
    """
    Elimina filas duplicadas del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a limpiar
        subset (list, optional): Columnas a considerar para identificar duplicados
        keep (str): 'first', 'last' o False para eliminar todos los duplicados
    
    Returns:
        pd.DataFrame: DataFrame sin duplicados
    
    Example:
        >>> df_clean = remove_duplicates(df, subset=['id', 'name'])
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    removed = initial_rows - len(df_clean)
    print(f"Duplicados eliminados: {removed} filas")
    return df_clean


def handle_missing_values(df, strategy='drop', columns=None, fill_value=None):
    """
    Maneja valores faltantes (NaN) en el DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a limpiar
        strategy (str): Estrategia a usar:
            - 'drop': Eliminar filas con NaN
            - 'mean': Rellenar con la media (solo numéricas)
            - 'median': Rellenar con la mediana (solo numéricas)
            - 'mode': Rellenar con la moda
            - 'fill': Rellenar con valor específico
        columns (list, optional): Columnas a procesar (None = todas)
        fill_value: Valor para usar con strategy='fill'
    
    Returns:
        pd.DataFrame: DataFrame con NaN tratados
    
    Example:
        >>> df_clean = handle_missing_values(df, strategy='median', columns=['salary'])
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        null_count = df_clean[col].isnull().sum()
        if null_count == 0:
            continue
        
        if strategy == 'drop':
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == 'mean':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == 'median':
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == 'mode':
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
        elif strategy == 'fill':
            df_clean[col] = df_clean[col].fillna(fill_value)
        
        print(f"Columna '{col}': {null_count} valores nulos tratados con '{strategy}'")
    
    return df_clean


def fix_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detecta y corrige valores atípicos (outliers) en una columna numérica.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar
        column (str): Nombre de la columna a analizar
        method (str): Método de detección:
            - 'iqr': Rango intercuartílico (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
            - 'zscore': Z-score (valores fuera de ±threshold desviaciones estándar)
            - 'cap': Capeo/Winsorización al percentil 1 y 99
        threshold (float): Umbral para el método (default 1.5 para IQR, 3 para zscore)
    
    Returns:
        pd.DataFrame: DataFrame con outliers corregidos
    
    Example:
        >>> df_clean = fix_outliers(df, 'salary_usd', method='iqr')
    """
    df_clean = df.copy()
    
    if not pd.api.types.is_numeric_dtype(df_clean[column]):
        print(f"Advertencia: '{column}' no es numérica. Saltando...")
        return df_clean
    
    original_outliers = 0
    
    if method == 'iqr':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
        original_outliers = outliers_mask.sum()
        
        df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
        df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
        
    elif method == 'zscore':
        mean = df_clean[column].mean()
        std = df_clean[column].std()
        z_scores = (df_clean[column] - mean) / std
        
        outliers_mask = abs(z_scores) > threshold
        original_outliers = outliers_mask.sum()
        
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        df_clean.loc[df_clean[column] < lower_bound, column] = lower_bound
        df_clean.loc[df_clean[column] > upper_bound, column] = upper_bound
        
    elif method == 'cap':
        lower_bound = df_clean[column].quantile(0.01)
        upper_bound = df_clean[column].quantile(0.99)
        
        outliers_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)
        original_outliers = outliers_mask.sum()
        
        df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"Columna '{column}': {original_outliers} outliers corregidos con método '{method}'")
    return df_clean


def normalize_strings(df, column, operations=None):
    """
    Normaliza valores de texto en una columna.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar
        column (str): Nombre de la columna a normalizar
        operations (list): Lista de operaciones a aplicar:
            - 'lower': Convertir a minúsculas
            - 'upper': Convertir a mayúsculas
            - 'title': Capitalizar primera letra de cada palabra
            - 'strip': Eliminar espacios al inicio y final
            - 'remove_extra_spaces': Eliminar espacios múltiples
    
    Returns:
        pd.DataFrame: DataFrame con strings normalizados
    
    Example:
        >>> df_clean = normalize_strings(df, 'country', ['strip', 'title'])
    """
    if operations is None:
        operations = ['strip', 'lower']
    
    df_clean = df.copy()
    
    # Asegurar que la columna es string
    df_clean[column] = df_clean[column].astype(str)
    
    for op in operations:
        if op == 'lower':
            df_clean[column] = df_clean[column].str.lower()
        elif op == 'upper':
            df_clean[column] = df_clean[column].str.upper()
        elif op == 'title':
            df_clean[column] = df_clean[column].str.title()
        elif op == 'strip':
            df_clean[column] = df_clean[column].str.strip()
        elif op == 'remove_extra_spaces':
            df_clean[column] = df_clean[column].str.replace(r'\s+', ' ', regex=True)
    
    print(f"Columna '{column}': strings normalizados con {operations}")
    return df_clean


def standardize_categories(df, column, mapping):
    """
    Estandariza valores categóricos usando un diccionario de mapeo.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar
        column (str): Nombre de la columna a estandarizar
        mapping (dict): Diccionario de mapeo {valor_original: valor_estandar}
    
    Returns:
        pd.DataFrame: DataFrame con categorías estandarizadas
    
    Example:
        >>> mapping = {'usa': 'USA', 'united states': 'USA', 'u.s.a.': 'USA'}
        >>> df_clean = standardize_categories(df, 'country', mapping)
    """
    df_clean = df.copy()
    
    # Crear mapeo case-insensitive
    lower_mapping = {k.lower(): v for k, v in mapping.items()}
    
    original_values = df_clean[column].nunique()
    df_clean[column] = df_clean[column].str.lower().str.strip().map(lower_mapping).fillna(df_clean[column])
    new_values = df_clean[column].nunique()
    
    print(f"Columna '{column}': categorías reducidas de {original_values} a {new_values}")
    return df_clean


def convert_to_numeric(df, column, errors='coerce'):
    """
    Convierte una columna a tipo numérico, manejando valores no numéricos.
    
    Args:
        df (pd.DataFrame): DataFrame a procesar
        column (str): Nombre de la columna a convertir
        errors (str): Cómo manejar errores:
            - 'coerce': Convertir errores a NaN
            - 'ignore': Mantener valores originales
            - 'raise': Lanzar excepción
    
    Returns:
        pd.DataFrame: DataFrame con columna convertida
    
    Example:
        >>> df_clean = convert_to_numeric(df, 'salary_usd')
    """
    df_clean = df.copy()
    
    non_numeric = df_clean[column].apply(lambda x: not str(x).replace('.', '').replace('-', '').isdigit() if pd.notna(x) else False).sum()
    
    df_clean[column] = pd.to_numeric(df_clean[column], errors=errors)
    
    print(f"Columna '{column}': convertida a numérico ({non_numeric} valores no numéricos encontrados)")
    return df_clean
