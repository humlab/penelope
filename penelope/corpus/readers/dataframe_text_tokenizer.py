from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
from penelope.vendor.nltk import word_tokenize

from .interfaces import ICorpusReader
from .text_transformer import TextTransformer


class DataFrameTextTokenizer(
    ICorpusReader
):  # pylint: disable=too-many-instance-attributes, disable=too-many-return-statements
    """Text iterator that returns row-wise text documents from a Pandas DataFrame"""

    def __init__(self, data: pd.DataFrame, text_column='txt', **column_filters):
        """
        Parameters
        ----------
        df : DataFrame
            Data frame having one document per row. Text must be in column `text_column` and filename/id in `filename`
        """
        assert text_column in data.columns

        self.data = data
        self.text_column = text_column

        if 'filename' not in self.data.columns:
            self.data['filename'] = self.data.index.astype(str)

        for column, value in column_filters.items():
            assert column in self.data.columns, column + ' is missing'
            if isinstance(value, tuple):
                assert len(value) == 2
                self.data = self.data[self.data[column].between(*value)]
            elif isinstance(value, list):
                self.data = self.data[self.data[column].isin(value)]
            else:
                self.data = self.data[self.data[column] == value]

        if len(self.data[self.data[text_column].isna()]) > 0:
            print('Warn: {} n/a rows encountered'.format(len(self.data[self.data[text_column].isna()])))
            self.data = self.data.dropna()

        self.text_transformer = TextTransformer(transforms=[]).fix_unicode().fix_whitespaces().fix_hyphenation()

        self.iterator = None
        self._metadata = self.data.drop(self.text_column, axis=1).to_dict(orient='records')
        self._documents = pd.DataFrame(self._metadata)
        self._filenames = [x['filename'] for x in self._metadata]
        self.tokenize = word_tokenize

    def _create_iterator(self) -> Iterator[Tuple[str, List[str]]]:
        return (self._process(row['filename'], row[self.text_column]) for _, row in self.data.iterrows())

    def _process(self, filename: str, text: str) -> Tuple[str, List[str]]:
        """Process the text and returns tokenized text"""
        # text = self.preprocess(text)

        text = self.text_transformer.transform(text)

        tokens = self.tokenize(
            text,
        )

        return filename, tokens

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self._create_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise

    @property
    def filenames(self) -> List[str]:
        return self._filenames

    @property
    def metadata(self) -> List[Dict[str, Any]]:
        return self._metadata

    @property
    def documents(self) -> pd.DataFrame:
        return self._documents
