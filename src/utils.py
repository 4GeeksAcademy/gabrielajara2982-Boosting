"""
utils.py
Funciones para análisis exploratorio de datos (EDA)
Autor: [Gabriela Jara]
"""

from dotenv import load_dotenv
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load the .env file variables
load_dotenv()


def db_connect():
    import os
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine


# ==============================
# FUNCIONES DE INSPECCIÓN GENERAL
# ==============================

def resumen_datos(df: pd.DataFrame):
    """
    Muestra información general del dataset:
    - Dimensiones
    - Tipos de datos
    - Valores nulos
    - Estadísticas descriptivas
    """
    print("="*60)
    print(f"Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("="*60)
    print("\nTipos de datos:")
    print(df.dtypes)
    print("="*60)
    print("\nValores nulos por columna:")
    print(df.isnull().sum())
    print("="*60)
    print("\nEstadísticas descriptivas:")
    display(df.describe(include='all').T)


# ==============================
# FUNCIONES DE ANÁLISIS UNIVARIADO
# ==============================

def plot_distribucion(df: pd.DataFrame, col: str, bins: int = 30):
    """
    Dibuja la distribución de una variable numérica.
    """
    plt.figure(figsize=(7,4))
    sns.histplot(df[col].dropna(), kde=True, bins=bins)
    plt.title(f"Distribución de {col}", fontsize=12)
    plt.xlabel(col)
    plt.ylabel("Frecuencia")
    plt.show()


def plot_categorica(df: pd.DataFrame, col: str):
    """
    Dibuja un gráfico de barras para una variable categórica.
    """
    plt.figure(figsize=(7,4))
    order = df[col].value_counts().index
    sns.countplot(x=df[col], order=order)
    plt.title(f"Distribución de {col}", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()


# ==============================
# FUNCIONES DE ANÁLISIS BIVARIADO
# ==============================

def correlacion_num(df: pd.DataFrame):
    """
    Muestra un mapa de calor con la matriz de correlación de variables numéricas.
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de correlación")
    plt.show()


def boxplot_vs_target(df: pd.DataFrame, feature: str, target: str):
    """
    Boxplot para analizar la relación entre una variable categórica y una numérica.
    """
    plt.figure(figsize=(8,5))
    sns.boxplot(x=feature, y=target, data=df)
    plt.title(f"{target} vs {feature}")
    plt.xticks(rotation=45)
    plt.show()


# ==============================
# FUNCIONES DE VALORES FALTANTES Y OUTLIERS
# ==============================

def porcentaje_nulos(df: pd.DataFrame):
    """
    Devuelve un DataFrame con el porcentaje de valores nulos por columna.
    """
    nulos = df.isnull().mean().sort_values(ascending=False) * 100
    return nulos[nulos > 0].to_frame("Porcentaje de Nulos (%)")


def detectar_outliers_iqr(df: pd.DataFrame, col: str):
    """
    Detecta outliers usando el método IQR para una columna numérica.
    Retorna un DataFrame con los outliers encontrados.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
    print(f"Outliers detectados en {col}: {outliers.shape[0]}")
    return outliers


# ==============================
# FUNCIONES DE APOYO
# ==============================

def info_duplicados(df: pd.DataFrame):
    """
    Muestra cuántos registros duplicados existen en el DataFrame.
    """
    duplicados = df.duplicated().sum()
    print(f"Registros duplicados: {duplicados}")
    return duplicados
