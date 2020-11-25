from typing import Dict, Sequence

import pandas as pd
from penelope.utility import (
    IndexOfSplitOrCallableOrRegExp,
    extract_filenames_metadata,
    list_of_dicts_to_dict_of_lists,
    tuple_of_lists_to_list_of_tuples,
)

from .interfaces import ITokenizedCorpus
from .tokenized_corpus import ReiterableTerms


class SimpleTextLinesCorpus(ITokenizedCorpus):
    """Corpus that reads a document-per-line text file """

    def __init__(
        self,
        filename: str,
        fields: Dict[str, int],
        filename_fields: Sequence[IndexOfSplitOrCallableOrRegExp] = None,
        filename_fields_key: str = None,
        sep: str = ' # ',
    ):
        """Simple corpus for document per line data  """
        with open(filename, 'r') as f:
            lines = f.readlines()

        if 'filename' not in fields or 'text' not in fields:
            raise ValueError("Fields `filename` and `text` are not specified (required fields)")

        data = list_of_dicts_to_dict_of_lists(
            [{k: data[fields[k]] for k in fields} for data in [line.split(sep) for line in lines]]
        )

        self._filenames = data['filename']
        self.iterator = None
        self.tokens = [[x.lower() for x in text.split() if len(x) > 0] for text in data['text']]

        fields_data = {k: v for k, v in data.items() if k != 'text'}

        if filename_fields is not None:

            filename_data = extract_filenames_metadata(filenames=self._filenames, filename_fields=filename_fields)
            fields_data = {**fields_data, **list_of_dicts_to_dict_of_lists(filename_data)}

        self._documents = pd.DataFrame(data=fields_data)

        if filename_fields_key is not None:
            self._documents.set_index(filename_fields_key)
            self._documents[filename_fields_key] = self._documents.index

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
        return tuple_of_lists_to_list_of_tuples((self._filenames, self.tokens))

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
