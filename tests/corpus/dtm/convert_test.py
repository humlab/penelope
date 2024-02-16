from typing import Iterable, Tuple

import pandas as pd
import pytest
import scipy.sparse as sp
from pytest import fixture

from penelope import corpus as pc
from penelope.corpus.dtm import convert
from penelope.corpus.readers import tng
from penelope.vendor.gensim_api import GENSIM_INSTALLED
from penelope.vendor.gensim_api import corpora as gensim_corpora
from penelope.vendor.textacy_api import TEXTACY_INSTALLED

# pylint: disable=redefined-outer-name

SIMPLE_CORPUS_ABC_5DOCS = [
    ('d_01.txt', ['a', 'b', 'c', 'c']),
    ('d_02.txt', ['a', 'a', 'b']),
    ('d_03.txt', ['a', 'b']),
    ('d_04.txt', ['c', 'c', 'a']),
    ('d_05.txt', ['a', 'b', 'b', 'c']),
]

SIMPLE_BOW = [
    [(0, 1), (1, 1), (2, 2)],
    [(0, 2), (1, 1)],
    [(0, 1), (1, 1)],
    [(0, 1), (2, 2)],
    [(0, 1), (1, 2), (2, 1)],
]


EXPECTED_DENSE_VALUES = [[1, 1, 2], [2, 1, 0], [1, 1, 0], [1, 0, 2], [1, 2, 1]]


@fixture
def document_index() -> pd.DataFrame:
    document_names = [f'd_0{i}' for i in range(1, 6)]
    document_ids = [0, 1, 2, 3, 4]
    di = pd.DataFrame(
        data={
            'filename': document_names,
            'document_name': document_names,
            'document_id': document_ids,
            'year': 2022,
        },
        index=document_names,
    )
    return di


@fixture
def sparse() -> sp.spmatrix:
    return gensim_corpora.corpus2csc(SIMPLE_BOW)


@fixture
def token2id() -> dict:
    return {'a': 0, 'b': 1, 'c': 2}


@fixture
def tokenized_corpus():
    reader = tng.CorpusReader(source=tng.InMemorySource(SIMPLE_CORPUS_ABC_5DOCS), transformer=None)  # already tokenized
    corpus = pc.TokenizedCorpus(reader=reader)
    return corpus


def test_id2token2token2id():
    assert pc.id2token2token2id({1: 'a', 2: 'b'}) == {'a': 1, 'b': 2}
    assert pc.id2token2token2id(pc.Token2Id({1: 'a', 2: 'b'})) == {'a': 1, 'b': 2}


@pytest.mark.skipif(not GENSIM_INSTALLED, reason="Gensim not installed")
def test_from_sparse2corpus(document_index, sparse, token2id):
    source: gensim_corpora.Sparse2Corpus = gensim_corpora.Sparse2Corpus(sparse, documents_columns=True)
    corpus: pc.VectorizedCorpus = convert.from_sparse2corpus(
        source=source, token2id=token2id, document_index=document_index
    )
    assert corpus is not None


@pytest.mark.skipif(not GENSIM_INSTALLED, reason="Gensim not installed")
def test_from_spmatrix(document_index, sparse, token2id):
    source: sp.spmatrix = sparse.tocsr().T
    corpus: pc.VectorizedCorpus = convert.from_spmatrix(source=source, token2id=token2id, document_index=document_index)
    assert corpus is not None
    assert corpus.shape == (5, 3)
    assert corpus.data.astype(int).todense().tolist() == EXPECTED_DENSE_VALUES


@pytest.mark.skip(reason="Not implemented")
def test_from_tokenized_corpus(document_index):
    source: pc.TokenizedCorpus = None
    vectorize_opts: pc.VectorizeOpts = pc.VectorizeOpts(already_tokenized=True)
    corpus: pc.VectorizedCorpus = convert.from_tokenized_corpus(
        source=source, document_index=document_index, vectorize_opts=vectorize_opts
    )
    assert corpus is not None
    assert corpus.shape == (5, 3)
    assert corpus.data.astype(int).todense().tolist() == EXPECTED_DENSE_VALUES


@pytest.mark.skipif(not TEXTACY_INSTALLED, reason="Textacy not installed")
def test_from_stream_of_tokens(document_index, token2id):
    source: Iterable[Iterable[str]] = [x[1] for x in SIMPLE_CORPUS_ABC_5DOCS]
    vectorize_opts: pc.VectorizeOpts = pc.VectorizeOpts(already_tokenized=True)
    corpus: pc.VectorizedCorpus = convert.from_stream_of_tokens(
        source=source, token2id=token2id, document_index=document_index, vectorize_opts=vectorize_opts
    )
    assert corpus is not None
    assert corpus.shape == (5, 3)
    assert corpus.data.astype(int).todense().tolist() == EXPECTED_DENSE_VALUES


def test_from_stream_of_filename_tokens(document_index, token2id):
    source: Iterable[Tuple[str, Iterable[str]]] = SIMPLE_CORPUS_ABC_5DOCS
    vectorize_opts: pc.VectorizeOpts = pc.VectorizeOpts(already_tokenized=True)
    corpus: pc.VectorizedCorpus = convert.from_stream_of_filename_tokens(
        source=source, token2id=token2id, document_index=document_index, vectorize_opts=vectorize_opts
    )
    assert corpus is not None
    assert corpus.shape == (5, 3)
    assert corpus.data.astype(int).todense().tolist() == EXPECTED_DENSE_VALUES


def test_from_stream_of_text(document_index, token2id):
    source: Iterable[Tuple[str, Iterable[str]]] = [(x[0], ' '.join(x[1])) for x in SIMPLE_CORPUS_ABC_5DOCS]
    vectorize_opts: pc.VectorizeOpts = pc.VectorizeOpts(already_tokenized=False)
    corpus: pc.VectorizedCorpus = convert.from_stream_of_text(
        source=source, token2id=token2id, document_index=document_index, vectorize_opts=vectorize_opts
    )
    assert corpus is not None
    assert corpus.shape == (5, 3)
    assert corpus.data.astype(int).todense().tolist() == EXPECTED_DENSE_VALUES


def test_translate(): ...
