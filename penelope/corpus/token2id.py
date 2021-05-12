import pathlib
from collections import defaultdict
from collections.abc import MutableMapping
from typing import Iterator, Optional, Union

import pandas as pd
from loguru import logger
from penelope.utility import pandas_to_csv_zip, strip_paths


class Token2Id(MutableMapping):
    """A token-to-id mapping (dictionary)"""

    def __init__(self, data: Optional[Union[dict, defaultdict]] = None):
        if isinstance(data, defaultdict):
            self.data = data
        elif isinstance(data, dict):
            self.data = defaultdict(int, data)
        else:
            self.data = data or defaultdict()
        self.data.default_factory = self.data.__len__

    def __getitem__(self, key):
        return self.data[self._keytransform(key)]

    def __setitem__(self, key, value):
        self.data[self._keytransform(key)] = value

    def __delitem__(self, key):
        del self.data[self._keytransform(key)]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def _keytransform(self, key):
        return key

    def ingest(self, tokens: Iterator[str]) -> "Token2Id":
        for token in tokens:
            _ = self.data[token]
        return self

    def close(self) -> "Token2Id":
        self.data.default_factory = None

    def open(self) -> "Token2Id":
        self.data.default_factory = self.__len__
        return self

    def id2token(self) -> dict:
        return {v: k for k, v in self.data.items()}

    def to_dataframe(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame({'token': self.data.keys(), 'token_id': self.data.values()}).set_index('token')
        return df

    def store(self, filename: str):
        """Store dictionary as CSV"""
        pandas_to_csv_zip(filename, dfs=(self.to_dataframe(), strip_paths(filename)), sep='\t', header=True)

    @staticmethod
    def load(filename: str) -> "Token2Id":
        """Store dictionary as CSV"""
        if not pathlib.Path(filename).exists():
            logger.info("bundle has no vocabulary")
            return None
        df: pd.DataFrame = pd.read_csv(filename, sep='\t', index_col=0)
        data: dict = df['token_id'].to_dict()
        return Token2Id(data=data)
