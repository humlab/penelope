import os
from typing import Any, Callable, List

import numpy as np
import pandas as pd
import penelope.topic_modelling as topic_modelling
from penelope.co_occurrence import Bundle, to_filename
from penelope.corpus import TextTransformOpts, VectorizedCorpus
from penelope.corpus.readers import TextReader, TextReaderOpts, TextTokenizer

from .fixtures import TranströmerCorpus

OUTPUT_FOLDER = './tests/output'
TEST_DATA_FOLDER = './tests/test_data'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TEST_CORPUS_FILENAME = os.path.join(TEST_DATA_FOLDER, 'test_corpus.zip')
TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME = os.path.join(TEST_DATA_FOLDER, 'tranströmer_corpus_export.sparv4.csv.zip')
PERSISTED_INFERRED_MODEL_SOURCE_FOLDER: str = './tests/test_data/tranströmer_inferred_model'

# pylint: disable=too-many-arguments

TOPIC_MODELING_OPTS = {
    'n_topics': 4,
    'passes': 1,
    'random_seed': 42,
    'alpha': 'auto',
    'workers': 1,
    'max_iter': 100,
    'prefix': '',
}

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)

# http://www.nltk.org/howto/collocations.html
# PMI


class incline_code:
    def __init__(self, source: Any, msg: str = ""):
        self.source: str = source
        self.msg: str = msg

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


def create_abc_corpus(dtm: List[List[int]], document_years: List[int] = None) -> VectorizedCorpus:

    bag_term_matrix = np.array(dtm)
    token2id = {chr(ord('a') + i): i for i in range(0, bag_term_matrix.shape[1])}

    years: List[int] = (
        document_years if document_years is not None else [2000 + i for i in range(0, bag_term_matrix.shape[0])]
    )

    document_index = pd.DataFrame(
        {
            'year': years,
            'filename': [f'{2000+i}_{i}.txt' for i in years],
            'document_id': [i for i in range(0, bag_term_matrix.shape[0])],
        }
    )
    corpus: VectorizedCorpus = VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return corpus


def create_vectorized_corpus() -> VectorizedCorpus:
    return create_abc_corpus(
        dtm=[
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [2, 4, 1, 1],
            [2, 0, 1, 1],
        ],
        document_years=[2013, 2013, 2014, 2014, 2014],
    )


def create_text_reader(
    source_path=TEST_CORPUS_FILENAME,
    as_binary: bool = False,
    filename_fields=None,
    index_field=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    fix_hyphenation: bool = True,
    fix_whitespaces: bool = False,
) -> TextReader:
    reader_opts = TextReaderOpts(
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        filename_fields=filename_fields,
        index_field=index_field,
        as_binary=as_binary,
    )
    transform_opts = TextTransformOpts(fix_whitespaces=fix_whitespaces, fix_hyphenation=fix_hyphenation)
    reader = TextReader(source=source_path, reader_opts=reader_opts, transform_opts=transform_opts)
    return reader


def create_tokens_reader(
    source_path=TEST_CORPUS_FILENAME,
    as_binary: bool = False,
    filename_fields=None,
    index_field=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    fix_hyphenation: bool = True,
    fix_whitespaces: bool = False,
    chunk_size: int = None,
    tokenize: Callable = None,
) -> TextTokenizer:
    reader_opts = TextReaderOpts(
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        filename_fields=filename_fields,
        index_field=index_field,
        as_binary=as_binary,
    )
    transform_opts = TextTransformOpts(fix_whitespaces=fix_whitespaces, fix_hyphenation=fix_hyphenation)
    reader = TextTokenizer(
        source_path,
        reader_opts=reader_opts,
        transform_opts=transform_opts,
        tokenize=tokenize,
        chunk_size=chunk_size,
    )
    return reader


def create_inferred_model(method="gensim_lda-multicore") -> topic_modelling.InferredModel:

    corpus: TranströmerCorpus = TranströmerCorpus()
    train_corpus: topic_modelling.TrainingCorpus = topic_modelling.TrainingCorpus(
        terms=corpus.terms,
        document_index=corpus.document_index,
    )

    inferred_model: topic_modelling.InferredModel = topic_modelling.infer_model(
        train_corpus=train_corpus,
        method=method,
        engine_args=TOPIC_MODELING_OPTS,
    )
    return inferred_model


def create_bundle(tag: str = 'DUMMY') -> Bundle:
    folder = f'./tests/test_data/{tag}'
    filename = to_filename(folder=folder, tag=tag)
    bundle: Bundle = Bundle.load(filename, compute_frame=False)
    return bundle
