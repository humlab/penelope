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
        self._id2token: dict = None

    def __getitem__(self, key):
        # return self.data[self._keytransform(key)]
        return self.data[key]

    def __setitem__(self, key, value):
        if self._id2token:
            self._id2token = None
        # self.data[self._keytransform(key)] = value
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]
        # del self.data[self._keytransform(key)]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    # FIXME Reason for this functioN? Remove if not used.
    # def _keytransform(self, key: str) -> str:
    #     return key

    def ingest(self, tokens: Iterator[str]) -> "Token2Id":
        self._id2token = None
        for token in tokens:
            _ = self.data[token]
        return self

    def close(self) -> "Token2Id":
        self.data.default_factory = None

    def open(self) -> "Token2Id":
        self.data.default_factory = self.__len__
        self._id2token = None
        return self

    def id2token(self) -> dict:
        if self._id2token is None:
            self._id2token = {v: k for k, v in self.data.items()}
        return self._id2token

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
