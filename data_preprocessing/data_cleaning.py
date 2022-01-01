from enum import Enum
from typing import Optional

from data_preprocessing.data_insights import *


class WrongValueNumericRule(Enum):
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    EUCLIDEAN = 0
    MANHATTAN = 1


def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    df = df.copy()
    if must_be_rule == WrongValueNumericRule.MUST_BE_LESS_THAN:
        for colValue in df[column].to_numpy():
            if colValue >= must_be_rule_optional_parameter:
                df[column].replace({int(colValue): np.nan}, inplace=True)

    if must_be_rule == WrongValueNumericRule.MUST_BE_GREATER_THAN:
        for colValue in df[column].to_numpy():
            if colValue <= must_be_rule_optional_parameter:
                df[column].replace({int(colValue): np.nan}, inplace=True)

    if must_be_rule == WrongValueNumericRule.MUST_BE_POSITIVE:
        for colValue in df[column].to_numpy():
            if colValue <= must_be_rule.value:
                df[column].replace({int(colValue): np.nan}, inplace=True)

    if must_be_rule == WrongValueNumericRule.MUST_BE_NEGATIVE:
        for colValue in df[column].to_numpy():
            if colValue >= must_be_rule.value:
                df[column].replace({int(colValue): np.nan}, inplace=True)
    return df


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in get_numeric_columns(df) and not (column in get_binary_columns(df)):
        tmp = []
        mean, std = df[column].mean(), df[column].std()
        lowerboundry, upperboundry = mean - std * 3, mean + std * 3
        for value in df[column]:
            if value < lowerboundry or value > upperboundry:
                tmp.append(int(mean))
            else:
                tmp.append(value)
        df[column] = tmp
    if column in get_text_categorical_columns(df):
        for record in df[column].to_numpy():
            if not (isinstance(record, str) or record is None):
                df[column].replace({record: None}, inplace=True)
    return df


def fix_nans(df: pd.DataFrame, column: str, replace_with=np.nan) -> pd.DataFrame:
    if column in get_binary_columns(df):
        df[column].replace({np.nan: df[column].mode()[0]}, inplace=True)
    else:
        if df[column].dtype.name == "object" or df[column].dtype.name == "category":
            for record in df[column].to_numpy():
                if record is None:
                    df[column].replace({record: df[column].mode()[0]}, inplace=True)
        else:
            if column in get_numeric_columns(df):
                get_column_count_of_nan(df, column)
                df[column].replace({np.nan: replace_with}, inplace=True)
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    pdseries = (df_column - df_column.min()) / (df_column.max() - df_column.min())
    return pdseries


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})

    assert fix_outliers(df, 'a') is not None
    assert fix_outliers(df, 'c') is not None
    assert fix_nans(df, 'c') is not None
    assert normalize_column(df.loc[:, 'a']) is not None
