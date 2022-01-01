from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


def pandas_profile(df: pd.DataFrame, result_html: str = 'report.html'):
    from pandas_profiling import ProfileReport
    profile = ProfileReport(df, title="Pandas Profiling Report")
    if result_html is not None:
        profile.to_file(result_html)
    return profile.to_json()


def get_column_max(df: pd.DataFrame, column_name: str) -> float:
    return df[column_name].max()


def get_column_min(df: pd.DataFrame, column_name: str) -> float:
    return df[column_name].min()


def get_column_mean(df: pd.DataFrame, column_name: str) -> float:
    mean = df[column_name].mean()
    return mean


def get_column_count_of_nan(df: pd.DataFrame, column_name: str) -> float:
    return df[column_name].isnull().sum()


def get_column_number_of_duplicates(df: pd.DataFrame, column_name: str) -> float:
    noduplicates = {}
    for record in df[column_name].to_numpy():
        noduplicates.update({record: 1})
    numberOfDuplicates = len(df[column_name]) - len(noduplicates)
    # numberOfDuplicates = len(df[column_name])-len(df[column_name].unique())
    return numberOfDuplicates


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    numericColumns = []
    for columnnames in range(len(df.columns)):
        if (np.issubdtype(df[df.columns[columnnames]], np.number)):
            numericColumns.append(df.columns[columnnames])
    return numericColumns


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    binaryColumnsList = []
    binaryColumnsSeries = df.columns[df.isin([0, 1]).all()]
    for binaryColumn in binaryColumnsSeries:
        binaryColumnsList.append(binaryColumn)
    return binaryColumnsList


def get_text_categorical_columns(df: pd.DataFrame) -> List[str]:
    categoricalColumns = []
    for columnnames in range(len(df.columns)):
        if ((df[df.columns[columnnames]].dtype.name) == "object"):
            categoricalColumns.append(df.columns[columnnames])
    return categoricalColumns


def get_correlation_between_columns(df: pd.DataFrame, col1: str, col2: str) -> float:
    correlation = df[col1].corr(df[col2])
    return correlation
