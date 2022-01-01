import pandas as pd
from pathlib import Path


def read_dataset(path: Path, delimiter=",") -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=delimiter)
    return df
