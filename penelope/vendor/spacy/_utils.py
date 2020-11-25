import zipfile
from io import StringIO
from typing import Iterable, Union

import pandas as pd


def to_text(data: Union[str, Iterable[str]]):
    return data if isinstance(data, str) else ' '.join(data)


def read_data_frame_from_zip(zf, filename):
    data_str = zf.read(filename).decode('utf-8')
    data_source = StringIO(data_str)
    df = pd.read_csv(data_source, sep='\t', index_col=0)
    return df


def write_data_frame_to_zip(df: pd.DataFrame, filename: str, zf: zipfile.ZipFile):
    assert isinstance(df, (pd.DataFrame,))
    data_str: str = df.to_csv(sep='\t', header=True)
    zf.writestr(filename, data=data_str)
