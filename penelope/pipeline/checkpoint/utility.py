import csv
import os
import pathlib
import zipfile
from io import StringIO
from typing import List

import numpy as np
import pandas as pd

from .interface import DOCUMENT_INDEX_FILENAME, FILE_PATTERN


def find_checkpoints(source_folder: str, file_pattern: str = FILE_PATTERN) -> List[str]:
    filenames = sorted(pathlib.Path(source_folder).rglob(file_pattern))
    return filenames


class EmptyCheckpointError(Exception):
    ...


class CorruptCheckpointError(Exception):
    ...


def read_document_index(archive_filename: str) -> pd.DataFrame:
    try:
        # FIXME: #123 Remove Parla-CLARIN fields
        with zipfile.ZipFile(archive_filename, mode="r") as fp:
            csv_str: str = fp.read(DOCUMENT_INDEX_FILENAME).decode("utf-8")
            df: pd.DataFrame = pd.read_csv(
                StringIO(csv_str),
                sep="\t",
                quoting=csv.QUOTE_NONE,
                index_col=0,
                dtype={
                    'speech_id': str,
                    'speaker': str,
                    'speech_date': str,
                    'speech_index': np.int16,
                    'document_name': str,
                    'filename': str,
                    'num_tokens': np.int32,
                    'num_words': np.int32,
                },
                engine='c',
            )

            return df
    except Exception as ex:
        if os.stat(archive_filename).st_size == 0:
            raise EmptyCheckpointError() from ex
        raise CorruptCheckpointError() from ex
