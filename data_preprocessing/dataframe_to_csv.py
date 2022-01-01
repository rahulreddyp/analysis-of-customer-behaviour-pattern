import pandas as pd
from pathlib import Path

df = pd.DataFrame()


def write_dataset(path: Path, df, encoding='utf-8', sep=','):
    df.to_csv(path, sep=sep, encoding=encoding)
    return None
