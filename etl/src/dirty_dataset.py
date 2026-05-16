"""
Script para crear un dataset 'sucio' a partir del dataset limpio.
Usa solo bibliotecas estándar de Python (sin pandas/numpy).

Impurezas introducidas:
- Valores NaN (~8% de celdas)
- Filas duplicadas (~3%)
- Outliers en variables numéricas (~2%)
- Inconsistencias de formato (~5%)
- Tipos de datos mezclados (~2%)
"""

import csv
import random

# Semilla para reproducibilidad
random.seed(42)


def load_csv(filepath):
    """Carga un archivo CSV y retorna headers y filas."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = [row for row in reader]
    return headers, rows


def save_csv(filepath, headers, rows):
    """Guarda headers y filas en un archivo CSV."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def get_column_index(headers, col_name):
    """Obtiene el índice de una columna por nombre."""
    return headers.index(col_name)


def introduce_nan_values(rows, headers, columns, percentage=0.08):
    """Introduce valores vacíos (NaN) en las columnas especificadas."""
    n_rows = len(rows)
    
    for col_name in columns:
        col_idx = get_column_index(headers, col_name)
        n_nan = int(n_rows * percentage)
        nan_indices = random.sample(range(n_rows), min(n_nan, n_rows))
        
        for idx in nan_indices:
            rows[idx][col_idx] = ''  # Valor vacío representa NaN
    
    return rows


def introduce_duplicates(rows, percentage=0.03):
    """Introduce filas duplicadas aleatorias."""
    n_duplicates = int(len(rows) * percentage)
    duplicate_indices = random.choices(range(len(rows)), k=n_duplicates)
    
    for idx in duplicate_indices:
        rows.append(rows[idx].copy())
    
    return rows


def introduce_outliers(rows, headers, percentage=0.02):
    """Introduce valores atípicos (outliers) en columnas numéricas."""
    n_rows = len(rows)
    
    outlier_configs = {
        'salary_usd': [(-50000, -1000), (600000, 1500000)],
        'experience': [(-10, -1), (55, 80)]
    }
    
    for col_name, ranges in outlier_configs.items():
        col_idx = get_column_index(headers, col_name)
        n_outliers = int(n_rows * percentage)
        outlier_indices = random.sample(range(n_rows), min(n_outliers, n_rows))
        
        for idx in outlier_indices:
            # Solo modificar si el valor actual es numérico
            try:
                current_val = rows[idx][col_idx]
                if current_val and current_val.lstrip('-').isdigit():
                    low_range, high_range = ranges
                    if random.choice([True, False]):
                        rows[idx][col_idx] = str(random.randint(low_range[0], low_range[1]))
                    else:
                        rows[idx][col_idx] = str(random.randint(high_range[0], high_range[1]))
            except (ValueError, IndexError):
                pass
    
    return rows


def introduce_format_inconsistencies(rows, headers, percentage=0.05):
    """Introduce inconsistencias de formato en columnas de texto."""
    n_rows = len(rows)
    
    country_variants = {
        'USA': ['usa', 'U.S.A.', 'United States', 'US', '  USA  ', 'UNITED STATES'],
        'UK': ['uk', 'U.K.', 'United Kingdom', 'GB', '  UK  ', 'UNITED KINGDOM'],
        'Germany': ['germany', 'GERMANY', 'DE', '  Germany  ', 'Deutschland'],
        'Canada': ['canada', 'CANADA', 'CA', '  Canada  '],
        'India': ['india', 'INDIA', 'IN', '  India  '],
        'France': ['france', 'FRANCE', 'FR', '  France  '],
        'Australia': ['australia', 'AUSTRALIA', 'AU', '  Australia  '],
        'Japan': ['japan', 'JAPAN', 'JP', '  Japan  '],
        'Brazil': ['brazil', 'BRAZIL', 'BR', '  Brazil  '],
        'Singapore': ['singapore', 'SINGAPORE', 'SG', '  Singapore  ']
    }
    
    education_variants = {
        'Bachelors': ['bachelors', 'BACHELORS', 'Bachelor', "Bachelor's", '  Bachelors  '],
        'Masters': ['masters', 'MASTERS', 'Master', "Master's", '  Masters  '],
        'PhD': ['phd', 'PHD', 'Ph.D.', 'Doctorate', '  PhD  '],
        'High School': ['high school', 'HIGH SCHOOL', 'HighSchool', 'Secondary', '  High School  '],
        'Some College': ['some college', 'SOME COLLEGE', 'SomeCollege', 'Incomplete', '  Some College  ']
    }
    
    n_changes = int(n_rows * percentage)
    
    # Países
    country_idx = get_column_index(headers, 'country')
    country_indices = random.sample(range(n_rows), min(n_changes, n_rows))
    for idx in country_indices:
        original = rows[idx][country_idx]
        if original in country_variants:
            rows[idx][country_idx] = random.choice(country_variants[original])
    
    # Educación
    education_idx = get_column_index(headers, 'education')
    education_indices = random.sample(range(n_rows), min(n_changes, n_rows))
    for idx in education_indices:
        original = rows[idx][education_idx]
        if original in education_variants:
            rows[idx][education_idx] = random.choice(education_variants[original])
    
    return rows


def introduce_mixed_types(rows, headers, columns, percentage=0.02):
    """Introduce valores de texto en columnas numéricas."""
    n_rows = len(rows)
    
    mixed_values = ['N/A', 'unknown', '-', 'null', 'NA', 'n/a', 'missing', '?', 'NaN', '']
    
    for col_name in columns:
        col_idx = get_column_index(headers, col_name)
        n_mixed = int(n_rows * percentage)
        mixed_indices = random.sample(range(n_rows), min(n_mixed, n_rows))
        
        for idx in mixed_indices:
            rows[idx][col_idx] = random.choice(mixed_values)
    
    return rows


def create_dirty_dataset(input_path, output_path):
    """Función principal que crea el dataset sucio aplicando todas las impurezas."""
    print(f"Cargando dataset desde: {input_path}")
    headers, rows = load_csv(input_path)
    print(f"Dataset original: {len(rows)} filas, {len(headers)} columnas")
    
    # 1. Introducir valores NaN
    print("\n1. Introduciendo valores NaN...")
    rows = introduce_nan_values(rows, headers, ['education', 'frameworks', 'languages'], percentage=0.08)
    rows = introduce_nan_values(rows, headers, ['salary_usd', 'experience'], percentage=0.05)
    
    # 2. Introducir duplicados
    print("2. Introduciendo filas duplicadas...")
    rows = introduce_duplicates(rows, percentage=0.03)
    
    # 3. Introducir outliers
    print("3. Introduciendo outliers...")
    rows = introduce_outliers(rows, headers, percentage=0.02)
    
    # 4. Introducir inconsistencias de formato
    print("4. Introduciendo inconsistencias de formato...")
    rows = introduce_format_inconsistencies(rows, headers, percentage=0.05)
    
    # 5. Introducir tipos mezclados
    print("5. Introduciendo tipos de datos mezclados...")
    rows = introduce_mixed_types(rows, headers, ['experience', 'salary_usd'], percentage=0.02)
    
    # Mezclar filas para que las impurezas no estén al final
    random.shuffle(rows)
    
    # Guardar dataset sucio
    print(f"\nGuardando dataset sucio en: {output_path}")
    save_csv(output_path, headers, rows)
    
    # Calcular estadísticas
    total_rows = len(rows)
    
    # Contar valores vacíos por columna
    print("\n" + "="*50)
    print("RESUMEN DEL DATASET SUCIO")
    print("="*50)
    print(f"Total de filas: {total_rows}")
    print(f"Total de columnas: {len(headers)}")
    
    print(f"\nValores vacíos/nulos por columna:")
    for i, col in enumerate(headers):
        empty_count = sum(1 for row in rows if row[i] == '' or row[i] in ['N/A', 'null', 'NA', 'n/a', 'missing', '?', 'NaN', 'unknown', '-'])
        print(f"  {col}: {empty_count} ({100*empty_count/total_rows:.1f}%)")
    
    # Contar duplicados
    row_tuples = [tuple(row) for row in rows]
    unique_rows = set(row_tuples)
    duplicates = len(row_tuples) - len(unique_rows)
    print(f"\nFilas duplicadas: {duplicates}")
    
    return headers, rows


if __name__ == "__main__":
    input_path = "/home/nicolas/U/PrograCien/parcial1/Software_Developer_Salary/test.csv"
    output_path = "/home/nicolas/U/PrograCien/parcial1/data/raw/software_developer_salary_raw.csv"
    
    headers, rows = create_dirty_dataset(input_path, output_path)
    print("\n¡Dataset sucio creado exitosamente!")
