from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from data_preprocessing.data_insights import *


def generate_label_encoder(df_column: pd.Series) -> LabelEncoder:
    labelEncoding = LabelEncoder()
    labelEncoding.fit(df_column)
    return labelEncoding


def generate_one_hot_encoder(df_column: pd.Series) -> OneHotEncoder:
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(df_column.values.reshape(-1, 1))
    return ohe


def replace_with_label_encoder(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    dff = df.copy()
    dff[column] = le.transform(df[column])
    return dff


def replace_with_one_hot_encoder(df: pd.DataFrame, column: str, ohe: OneHotEncoder,
                                 ohe_column_names: List[str]) -> pd.DataFrame:
    df = df.copy()
    onehoten = ohe.transform(df[column].values.reshape(-1, 1))
    row, columns = onehoten.shape
    for i in range(columns):
        tmp = []
        for j in range(row):
            tmp.append(onehoten[j][i])
        df[ohe_column_names[i]] = tmp
    df = df.drop(column, 1)
    return df


def replace_label_encoder_with_original_column(df: pd.DataFrame, column: str, le: LabelEncoder) -> pd.DataFrame:
    df = df.copy()
    df[column] = le.inverse_transform(df[column])
    return df


def replace_one_hot_encoder_with_original_column(df: pd.DataFrame,
                                                 columns: List[str],
                                                 ohe: OneHotEncoder,
                                                 original_column_name: str) -> pd.DataFrame:
    df = df.copy()
    df[original_column_name] = ohe.inverse_transform(df[columns])
    df = df.drop(columns, 1)
    return df


if __name__ == "__main__":
    df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [True, True, False, False], 'c': ['one', 'two', 'three', 'four']})
    le = generate_label_encoder(df.loc[:, 'c'])
    assert le is not None
    ohe = generate_one_hot_encoder(df.loc[:, 'c'])
    assert ohe is not None
    assert replace_with_label_encoder(df, 'c', le) is not None
    assert replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())) is not None
    assert replace_label_encoder_with_original_column(replace_with_label_encoder(df, 'c', le), 'c', le) is not None
    assert replace_one_hot_encoder_with_original_column(
        replace_with_one_hot_encoder(df, 'c', ohe, list(ohe.get_feature_names())),
        list(ohe.get_feature_names()),
        ohe,
        'c') is not None
