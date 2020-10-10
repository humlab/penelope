import types
from typing import List

import pandas as pd
import penelope.utility as utility


class SimpleTestCorpus:

    def __init__(self, filename, meta_fields: List[str] = None):

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.corpus_data = [
            types.SimpleNamespace(
                filename=data[0],
                title=data[1],
                text=data[2],
                tokens=[x.lower() for x in data[2].split() if len(x) > 0]
            ) for data in [line.split(' # ') for line in lines]
        ]
        self.filenames = [x.filename for x in self.corpus_data]
        self.iterator = None

        meta_data = {'filename': self.filenames, 'title': [x.title for x in self.corpus_data]}

        if meta_fields is not None:

            filename_fields = utility.filename_field_parser(meta_fields)
            filename_data = [
                utility.extract_filename_fields(filename, **filename_fields) for filename in self.filenames
            ]
            meta_data = {**meta_data, **{k: [getattr(v, k) for v in filename_data] for k in filename_fields.keys()}}

        self.documents = pd.DataFrame(data=meta_data)
        if 'document_id' not in self.documents.columns:
            self.documents['document_id'] = self.documents.index

    def _create_iterator(self):
        return ((x.filename, x.tokens) for x in self.corpus_data)

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
