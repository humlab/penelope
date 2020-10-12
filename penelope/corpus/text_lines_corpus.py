from typing import Dict, List

import pandas as pd

import penelope.utility as utility
from penelope.corpus.tokenized_corpus import ReIterableTerms

def list_of_dicts_to_dict_of_lists(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}

def dict_of_lists_to_list_of_dicts(dl):
    return [dict(zip(dl,t)) for t in zip(*dl.values())]

def tuple_of_lists_to_list_of_tuples(tl):
    return zip(*tl)

class SimpleTextLinesCorpus:

    """Corpus that reads a file with documents on a single text-line seperated by `sep`character sequence"""
    def __init__(self, filename: str, fields: Dict[str, int], meta_fields: List[str] = None, sep: str=' # '):

        with open(filename, 'r') as f:
            lines = f.readlines()

        if 'filename' not in fields or 'text' not in fields:
            raise ValueError("Fields `filename` and `text` are not specified (required fields)")

        corpus_data = list_of_dicts_to_dict_of_lists([ {
                k: data[fields[k]] for k in fields
            }
            for data in [line.split(sep) for line in lines]
        ])

        self.filenames = corpus_data['filename']

        self.iterator = None

        self.tokens = [ [x.lower() for x in text.split() if len(x) > 0] for text in corpus_data['text'] ]

        meta_data = { k: v for k,v in corpus_data.items()  if k not in ( 'text' ) }

        if meta_fields is not None:

            filename_fields = utility.filename_field_parser(meta_fields)
            filename_data = [
                utility.extract_filename_fields(filename, **filename_fields) for filename in self.filenames
            ]
            meta_data = {**meta_data, **{k: [getattr(v, k) for v in filename_data] for k in filename_fields.keys()}}

        self.documents = pd.DataFrame(data=meta_data)

        if 'document_id' not in self.documents.columns:
            self.documents['document_id'] = self.documents.index

    @property
    def terms(self):
        return ReIterableTerms(self)

    def _create_iterator(self):
        return tuple_of_lists_to_list_of_tuples((self.filenames, self.tokens))

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
