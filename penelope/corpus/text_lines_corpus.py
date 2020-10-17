from typing import Dict, List

import pandas as pd

import penelope.utility as utility

from .interfaces import ITokenizedCorpus
from .tokenized_corpus import ReiterableTerms


class SimpleTextLinesCorpus(ITokenizedCorpus):

    """Corpus that reads a file with documents on a single text-line seperated by `sep`character sequence"""

    def __init__(self, filename: str, fields: Dict[str, int], meta_fields: List[str] = None, sep: str = ' # '):

        with open(filename, 'r') as f:
            lines = f.readlines()

        if 'filename' not in fields or 'text' not in fields:
            raise ValueError("Fields `filename` and `text` are not specified (required fields)")

        data = utility.list_of_dicts_to_dict_of_lists(
            [{k: data[fields[k]] for k in fields} for data in [line.split(sep) for line in lines]]
        )

        self._filenames = data['filename']
        self.iterator = None
        self.tokens = [[x.lower() for x in text.split() if len(x) > 0] for text in data['text']]

        fields_data = {k: v for k, v in data.items() if k != 'text'}

        if meta_fields is not None:

            filename_fields = utility.filename_field_parser(meta_fields)
            filename_data = [
                utility.extract_filename_fields(filename, **filename_fields) for filename in self._filenames
            ]
            fields_data = {**fields_data, **utility.list_of_dicts_to_dict_of_lists(filename_data)}

        self._documents = pd.DataFrame(data=fields_data)

        if 'document_id' not in self._documents.columns:
            self._documents['document_id'] = self._documents.index

    @property
    def filenames(self):
        return self._filenames

    @property
    def documents(self):
        return self._documents

    @property
    def metadata(self):
        self.documents.to_dict('records')

    @property
    def terms(self):
        return ReiterableTerms(self)

    def _create_iterator(self):
        return utility.tuple_of_lists_to_list_of_tuples((self._filenames, self.tokens))

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
