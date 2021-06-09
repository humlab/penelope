import os
from typing import Callable

import penelope.topic_modelling as topic_modelling
from penelope.corpus import TextTransformOpts
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
