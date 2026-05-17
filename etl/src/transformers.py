"""
Módulo de transformadores personalizados.
Contiene clases y funciones para transformar datos usando sklearn.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Transformador para seleccionar columnas específicas de un DataFrame.
    
    Attributes:
        columns (list): Lista de nombres de columnas a seleccionar
    
    Example:
        >>> selector = ColumnSelector(['age', 'salary'])
        >>> X_selected = selector.fit_transform(df)
    """
    
    def __init__(self, columns):
        """
        Inicializa el selector con las columnas deseadas.
        
        Args:
            columns (list): Nombres de columnas a seleccionar
        """
        self.columns = columns
    
    def fit(self, X, y=None):
        """Ajusta el transformador (no hace nada en este caso)."""
        return self
    
    def transform(self, X):
        """
        Selecciona las columnas especificadas.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada
        
        Returns:
            pd.DataFrame: DataFrame con solo las columnas seleccionadas
        """
        return X[self.columns].copy()


class NumericImputer(BaseEstimator, TransformerMixin):
    """
    Transformador para imputar valores faltantes en columnas numéricas.
    
    Attributes:
        strategy (str): Estrategia de imputación ('mean', 'median', 'mode', 'constant')
        fill_value: Valor a usar cuando strategy='constant'
    
    Example:
        >>> imputer = NumericImputer(strategy='median')
        >>> X_imputed = imputer.fit_transform(df)
    """
    
    def __init__(self, strategy='mean', fill_value=0):
        """
        Inicializa el imputador.
        
        Args:
            strategy (str): Estrategia de imputación
            fill_value: Valor para strategy='constant'
        """
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = {}
    
    def fit(self, X, y=None):
        """
        Calcula las estadísticas para imputación.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada
            y: Ignorado
        
        Returns:
            self
        """
        X = pd.DataFrame(X)
        
        for col in X.columns:
            if self.strategy == 'mean':
                self.statistics_[col] = X[col].mean()
            elif self.strategy == 'median':
                self.statistics_[col] = X[col].median()
            elif self.strategy == 'mode':
                mode = X[col].mode()
                self.statistics_[col] = mode[0] if len(mode) > 0 else 0
            elif self.strategy == 'constant':
                self.statistics_[col] = self.fill_value
        
        return self
    
    def transform(self, X):
        """
        Aplica la imputación.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada
        
        Returns:
            pd.DataFrame: DataFrame con valores imputados
        """
        X = pd.DataFrame(X).copy()
        
        for col in X.columns:
            if col in self.statistics_:
                X[col] = X[col].fillna(self.statistics_[col])
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Transformador para codificar variables categóricas.
    
    Attributes:
        method (str): Método de codificación ('onehot', 'label', 'ordinal')
        handle_unknown (str): Cómo manejar categorías desconocidas
    
    Example:
        >>> encoder = CategoricalEncoder(method='onehot')
        >>> X_encoded = encoder.fit_transform(df[['country', 'education']])
    """
    
    def __init__(self, method='label', handle_unknown='ignore'):
        """
        Inicializa el codificador.
        
        Args:
            method (str): Método de codificación
            handle_unknown (str): Manejo de categorías desconocidas
        """
        self.method = method
        self.handle_unknown = handle_unknown
        self.encoders_ = {}
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """
        Ajusta los codificadores a los datos.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada
            y: Ignorado
        
        Returns:
            self
        """
        X = pd.DataFrame(X)
        
        for col in X.columns:
            if self.method == 'label':
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders_[col] = le
            elif self.method == 'onehot':
                ohe = OneHotEncoder(sparse_output=False, handle_unknown=self.handle_unknown)
                ohe.fit(X[[col]])
                self.encoders_[col] = ohe
        
        return self
    
    def transform(self, X):
        """
        Aplica la codificación.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada
        
        Returns:
            pd.DataFrame o np.array: Datos codificados
        """
        X = pd.DataFrame(X).copy()
        
        if self.method == 'label':
            for col in X.columns:
                if col in self.encoders_:
                    le = self.encoders_[col]
                    # Manejar valores desconocidos
                    X[col] = X[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            return X
        
        elif self.method == 'onehot':
            result_dfs = []
            for col in X.columns:
                if col in self.encoders_:
                    ohe = self.encoders_[col]
                    encoded = ohe.transform(X[[col]])
                    feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                    result_dfs.append(pd.DataFrame(encoded, columns=feature_names, index=X.index))
            
            return pd.concat(result_dfs, axis=1) if result_dfs else X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Transformador para detectar y manejar outliers usando IQR.
    
    Attributes:
        method (str): Método de manejo ('clip', 'remove', 'nan')
        threshold (float): Multiplicador del IQR (default 1.5)
    
    Example:
        >>> remover = OutlierRemover(method='clip', threshold=1.5)
        >>> X_clean = remover.fit_transform(df[['salary', 'experience']])
    """
    
    def __init__(self, method='clip', threshold=1.5):
        """
        Inicializa el removedor de outliers.
        
        Args:
            method (str): Método de manejo
            threshold (float): Multiplicador del IQR
        """
        self.method = method
        self.threshold = threshold
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        """
        Calcula los límites para detección de outliers.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada
            y: Ignorado
        
        Returns:
            self
        """
        X = pd.DataFrame(X)
        
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.threshold * IQR
            upper = Q3 + self.threshold * IQR
            self.bounds_[col] = (lower, upper)
        
        return self
    
    def transform(self, X):
        """
        Aplica el manejo de outliers.
        
        Args:
            X (pd.DataFrame): DataFrame de entrada
        
        Returns:
            pd.DataFrame: DataFrame con outliers manejados
        """
        X = pd.DataFrame(X).copy()
        
        for col in X.columns:
            if col in self.bounds_:
                lower, upper = self.bounds_[col]
                
                if self.method == 'clip':
                    X[col] = X[col].clip(lower=lower, upper=upper)
                elif self.method == 'nan':
                    X.loc[(X[col] < lower) | (X[col] > upper), col] = np.nan
                elif self.method == 'remove':
                    X = X[(X[col] >= lower) & (X[col] <= upper)]
        
        return X


def create_numeric_pipeline(scaler='standard'):
    """
    Crea un pipeline para procesar variables numéricas.
    
    Args:
        scaler (str): Tipo de escalador ('standard', 'minmax', None)
    
    Returns:
        sklearn.pipeline.Pipeline: Pipeline configurado
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    steps = [
        ('imputer', SimpleImputer(strategy='median'))
    ]
    
    if scaler == 'standard':
        steps.append(('scaler', StandardScaler()))
    elif scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    
    return Pipeline(steps)


def create_categorical_pipeline(encoding='onehot'):
    """
    Crea un pipeline para procesar variables categóricas.
    
    Args:
        encoding (str): Tipo de codificación ('onehot', 'label')
    
    Returns:
        sklearn.pipeline.Pipeline: Pipeline configurado
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    
    steps = [
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown'))
    ]
    
    if encoding == 'onehot':
        steps.append(('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore')))
    
    return Pipeline(steps)
