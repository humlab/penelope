import os

import numpy as np
import pandas as pd
import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as corpora
from penelope.co_occurrence import Bundle, to_filename
from penelope.corpus import VectorizedCorpus
from tests.utils import OUTPUT_FOLDER, create_tokens_reader

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def flatten(lst):
    return [x for ws in lst for x in ws]


def create_reader() -> readers.TextTokenizer:
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
    return reader


def create_corpus() -> corpora.TokenizedCorpus:
    reader = create_reader()
    transform_opts = corpora.TokensTransformOpts(
        only_any_alphanumeric=True,
        to_lower=True,
        remove_accents=False,
        min_len=2,
        max_len=None,
        keep_numerals=False,
    )
    corpus = corpora.TokenizedCorpus(reader, transform_opts=transform_opts)
    return corpus


def create_vectorized_corpus() -> VectorizedCorpus:
    bag_term_matrix = np.array(
        [
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [2, 4, 1, 1],
            [2, 0, 1, 1],
        ]
    )
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus: VectorizedCorpus = VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return v_corpus


def create_slice_by_n_count_test_corpus() -> VectorizedCorpus:
    bag_term_matrix = np.array([[1, 1, 4, 1], [0, 2, 3, 0], [0, 3, 2, 0], [0, 4, 1, 3], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    return VectorizedCorpus(bag_term_matrix, token2id, df)


def create_bundle(tag: str = 'VENUS') -> Bundle:
    folder = f'./tests/test_data/{tag}'
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle
