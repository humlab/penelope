import os
import shutil
import uuid
from io import StringIO
from os.path import isfile
from os.path import join as jj

import numpy as np
import pytest
from penelope.corpus import (
    CorpusVectorizer,
    TokenizedCorpus,
    TokensTransformOpts,
    VectorizedCorpus,
    load_metadata,
    store_metadata,
)
from penelope.corpus.document_index import load_document_index
from tests.utils import OUTPUT_FOLDER, create_tokens_reader

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=redefined-outer-name


@pytest.fixture
def text_corpus() -> TokenizedCorpus:
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
    transform_opts = TokensTransformOpts(
        only_any_alphanumeric=True,
        to_lower=True,
        remove_accents=False,
        min_len=2,
        max_len=None,
        keep_numerals=False,
    )
    corpus = TokenizedCorpus(reader, transform_opts=transform_opts)
    return corpus


@pytest.mark.parametrize('mode', ['bundle', 'files'])
def test_load_stored_metadata_simple(mode: str):

    tag: str = f'{uuid.uuid1()}'
    folder: str = jj(OUTPUT_FOLDER, tag)

    os.makedirs(folder, exist_ok=True)

    document_index_str: str = (
        ";filename;year;document_name;document_id\n"
        "a;a.txt;2019;a;0\n"
        "b;b.txt;2019;b;1\n"
        "c;c.txt;2019;c;2\n"
        "d;d.txt;2020;d;3\n"
        "e;e.txt;2020;e;4\n"
    )
    token2id: dict = dict(x=0, y=1, z=2)
    overridden_term_frequency = np.arange(3)

    metadata = {
        'document_index': load_document_index(filename=StringIO(document_index_str), sep=';'),
        'token2id': token2id,
        'overridden_term_frequency': overridden_term_frequency,
    }

    store_metadata(tag=tag, folder=folder, mode=mode, **metadata)

    metadata_loaded = load_metadata(tag=tag, folder=folder)
    assert metadata_loaded['document_index'].to_csv(sep=';') == document_index_str
    assert metadata_loaded['token2id'] == token2id
    assert (metadata_loaded['overridden_term_frequency'] == overridden_term_frequency).all()

    shutil.rmtree(folder)


@pytest.mark.parametrize('mode', ['bundle', 'files'])
def test_load_stored_metadata(mode: str, text_corpus: TokenizedCorpus):

    tag: str = f'{uuid.uuid1()}'
    folder: str = jj(OUTPUT_FOLDER, tag)

    os.makedirs(folder, exist_ok=True)

    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)

    store_metadata(tag=tag, folder=folder, mode=mode, **corpus.metadata)

    metadata = load_metadata(tag=tag, folder=folder)

    assert metadata is not None
    assert metadata['token2id'] == corpus.token2id
    assert (metadata['document_index'] == corpus.document_index).all().all()
    assert metadata['overridden_term_frequency'] == corpus.overridden_term_frequency

    shutil.rmtree(folder)


def test_load_dumped_corpus(text_corpus: TokenizedCorpus):

    tag: str = f'{uuid.uuid1()}'
    folder: str = jj(OUTPUT_FOLDER, tag)

    os.makedirs(folder, exist_ok=True)

    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)
    corpus.dump(tag=tag, folder=folder, compressed=True)

    assert isfile(jj(folder, f"{tag}_vectorizer_data.pickle"))

    loaded_corpus: VectorizedCorpus = VectorizedCorpus.load(tag=tag, folder=folder)

    assert (corpus.term_frequency == loaded_corpus.term_frequency).all()
    assert corpus.document_index.to_dict() == loaded_corpus.document_index.to_dict()
    assert corpus.token2id == loaded_corpus.token2id

    shutil.rmtree(folder)
