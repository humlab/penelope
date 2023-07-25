import contextlib
import glob
import os
import shutil
from typing import Any, Callable
import uuid

import numpy as np
import pandas as pd

from penelope.corpus import TextReaderOpts, TextTransformOpts, TokenizeTextReader, VectorizedCorpus

OUTPUT_FOLDER = './tests/output'
TEST_DATA_FOLDER = './tests/test_data'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=non-ascii-name
# pylint: disable=too-many-arguments

TEST_CORPUS_FILENAME = os.path.join(TEST_DATA_FOLDER, 'test_corpus.zip')
PERSISTED_INFERRED_MODEL_SOURCE_FOLDER: str = './tests/test_data/tranströmer/tranströmer_inferred_model'


if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)


def clear_output(path: str = './tests/output'):
    with contextlib.suppress(Exception):
        for f in glob.glob(path):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.unlink(f)


class inline_code:
    def __init__(self, source: Any, msg: str = ""):
        self.source: str = source
        self.msg: str = msg

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


class output_folder:
    def __init__(self, folder: str = None):
        self.folder: str = folder or str(uuid.uuid4())[:6]

    def __enter__(self):
        os.makedirs(self.folder, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.folder)


def create_abc_corpus(
    dtm: list[list[int]], document_years: list[int] = None, token2id: dict = None
) -> VectorizedCorpus:
    bag_term_matrix = np.array(dtm)
    token2id = token2id or {chr(ord('a') + i): i for i in range(0, bag_term_matrix.shape[1])}

    years: list[int] = (
        document_years if document_years is not None else [2000 + i for i in range(0, bag_term_matrix.shape[0])]
    )

    document_index = pd.DataFrame(
        {
            'year': years,
            'filename': [f'{2000+i}_{i}.txt' for i in years],
            'document_id': [i for i in range(0, bag_term_matrix.shape[0])],
        }
    )
    corpus: VectorizedCorpus = VectorizedCorpus(bag_term_matrix, token2id=token2id, document_index=document_index)
    return corpus


def simple_vectorized_abc_corpus() -> VectorizedCorpus:
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


def create_test_corpus_tokens_reader(
    source_path=TEST_CORPUS_FILENAME,
    as_binary: bool = False,
    filename_fields=None,
    index_field=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    text_transforms: str = "dehyphen,normalize-whitespace",
    chunk_size: int = None,
    tokenize: Callable = None,
) -> TokenizeTextReader:
    reader_opts = TextReaderOpts(
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        filename_fields=filename_fields,
        index_field=index_field,
        as_binary=as_binary,
    )
    transform_opts = TextTransformOpts(transforms=text_transforms)

    reader = TokenizeTextReader(
        source_path,
        reader_opts=reader_opts,
        transform_opts=transform_opts,
        tokenize=tokenize,
        chunk_size=chunk_size,
    )
    return reader
