import os

import numpy as np
import pandas as pd
import penelope.corpus.dtm as dtm
import pytest
from tests.utils import OUTPUT_FOLDER

from .utils import create_corpus, create_vectorized_corpus

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=redefined-outer-name


@pytest.fixture
def text_corpus() -> dtm.VectorizedCorpus:
    return create_corpus()


@pytest.fixture
def vectorized_corpus() -> dtm.VectorizedCorpus:
    return create_vectorized_corpus()


@pytest.fixture
def slice_corpus() -> dtm.VectorizedCorpus:
    return create_vectorized_corpus()


def test_vocabulary(vectorized_corpus):
    assert vectorized_corpus.vocabulary == ['a', 'b', 'c', 'd']


def test_term_frequencies(vectorized_corpus):
    assert vectorized_corpus.term_frequencies.tolist() == [10, 10, 11, 3]


def test_document_token_counts(vectorized_corpus):
    assert vectorized_corpus.document_token_counts.tolist() == [8, 7, 7, 8, 4]


def test_document_term_frequency_mapping(vectorized_corpus):
    assert vectorized_corpus.term_frequency_mapping == {'a': 10, 'b': 10, 'c': 11, 'd': 3}


def test_n_terms(vectorized_corpus):
    assert vectorized_corpus.n_terms == 4


def test_n_docs(vectorized_corpus):
    assert vectorized_corpus.n_docs == 5


def test_bag_term_matrix_to_bag_term_docs(vectorized_corpus):

    doc_ids = (
        0,
        1,
    )
    expected = [['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'], ['a', 'a', 'b', 'b', 'c', 'c', 'c']]
    docs = vectorized_corpus.to_bag_of_terms(doc_ids)
    assert expected == ([list(d) for d in docs])

    expected = [
        ['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'],
        ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'd'],
        ['a', 'a', 'c', 'd'],
    ]
    docs = vectorized_corpus.to_bag_of_terms()
    assert expected == ([list(d) for d in docs])


def test_load_of_uncompressed_corpus(text_corpus):

    # Arrange
    dumped_v_corpus: dtm.VectorizedCorpus = dtm.CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)

    dumped_v_corpus.dump(tag='dump_test', folder=OUTPUT_FOLDER, compressed=False)

    # Act
    loaded_v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus.load(tag='dump_test', folder=OUTPUT_FOLDER)

    # Assert
    assert dumped_v_corpus.term_frequency_mapping == loaded_v_corpus.term_frequency_mapping
    assert dumped_v_corpus.document_index.to_dict() == loaded_v_corpus.document_index.to_dict()
    assert dumped_v_corpus.token2id == loaded_v_corpus.token2id


def test_load_of_compressed_corpus(text_corpus):

    # Arrange
    dumped_v_corpus: dtm.VectorizedCorpus = dtm.CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)

    dumped_v_corpus.dump(tag='dump_test', folder=OUTPUT_FOLDER, compressed=True)

    # Act
    loaded_v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus.load(tag='dump_test', folder=OUTPUT_FOLDER)

    # Assert
    assert dumped_v_corpus.term_frequency_mapping == loaded_v_corpus.term_frequency_mapping
    assert dumped_v_corpus.document_index.to_dict() == loaded_v_corpus.document_index.to_dict()
    assert dumped_v_corpus.token2id == loaded_v_corpus.token2id


def test_id2token_is_reversed_token2id(vectorized_corpus):
    id2token = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    assert id2token == vectorized_corpus.id2token


def test_normalize_by_raw_counts():

    corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(
        bag_term_matrix=np.array([[4, 3, 7, 1], [6, 7, 4, 2]]),
        token2id={'a': 0, 'b': 1, 'c': 2, 'd': 3},
        document_index=pd.DataFrame({'year': [2013, 2014]}),
    )

    n_corpus = corpus.normalize()
    t_corpus = corpus.normalize_by_raw_counts()
    assert np.allclose(t_corpus.data.todense(), n_corpus.data.todense())


def test_dump_and_store_of_corpus_with_empty_trailing_row() -> dtm.VectorizedCorpus:

    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [0, 0, 0, 0]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014]})
    corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, document_index)

    corpus.dump(tag="ZERO", folder="./tests/output")

    loaded_corpus = dtm.VectorizedCorpus.load(tag="ZERO", folder="./tests/output")

    assert corpus.data.shape == loaded_corpus.data.shape


def test_find_matching_words_in_vocabulary():

    token2id = {"bengt": 0, "bertil": 1, "eva": 2, "julia": 3}

    assert dtm.find_matching_words_in_vocabulary(token2id, ["jens"]) == set()
    assert dtm.find_matching_words_in_vocabulary(token2id, []) == set()
    assert dtm.find_matching_words_in_vocabulary(token2id, ["bengt"]) == {"bengt"}
    assert dtm.find_matching_words_in_vocabulary(token2id, ["b*"]) == {"bengt", "bertil"}
    assert dtm.find_matching_words_in_vocabulary(token2id, ["|.*a|"]) == {"eva", "julia"}
    assert dtm.find_matching_words_in_vocabulary(token2id, ["*"]) == {"bengt", "bertil", "eva", "julia"}


def test_document_index(vectorized_corpus):
    assert vectorized_corpus.document_index is not None
    assert vectorized_corpus.document_index.columns.tolist() == ['year', 'n_raw_tokens']
    assert len(vectorized_corpus.document_index) == 5


def test_to_dense(vectorized_corpus):
    assert vectorized_corpus.todense() is not None


def test_get_word_vector(vectorized_corpus):
    assert vectorized_corpus.get_word_vector('b').tolist() == [1, 2, 3, 4, 0]


def test_filter(vectorized_corpus):
    assert len(vectorized_corpus.filter(lambda x: x['year'] == 2013).document_index) == 2


def test_n_global_top_tokens(vectorized_corpus):
    assert vectorized_corpus.n_global_top_tokens(2) == {'a': 10, 'c': 11}


def test_stats(vectorized_corpus):
    assert vectorized_corpus.stats() is not None


def test_to_n_top_dataframe(vectorized_corpus):
    assert vectorized_corpus.to_n_top_dataframe(1) is not None


def test_year_range(vectorized_corpus):
    assert vectorized_corpus.year_range() == (2013, 2014)


def test_xs_years(vectorized_corpus):
    assert vectorized_corpus.xs_years().tolist() == [2013, 2014]


def test_token_indices(vectorized_corpus):
    assert vectorized_corpus.token_indices(['a', 'c', 'z']) == [0, 2]


def test_tf_idf(vectorized_corpus):
    assert vectorized_corpus.tf_idf() is not None


def test_to_bag_of_terms(vectorized_corpus):
    expected_docs = [
        ['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'],
        ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'd'],
        ['a', 'a', 'c', 'd'],
    ]
    assert [list(x) for x in vectorized_corpus.to_bag_of_terms()] == expected_docs


def test_get_top_n_words(vectorized_corpus):
    assert vectorized_corpus.get_top_n_words(n=2) == [('c', 11), ('a', 10)]


def test_co_occurrence_matrix(vectorized_corpus):
    m = vectorized_corpus.co_occurrence_matrix()
    assert m is not None
    assert (
        m
        == np.matrix(
            [
                [0, 20, 22, 6],
                [0, 0, 20, 5],
                [0, 0, 0, 6],
                [0, 0, 0, 0],
            ]
        )
    ).all()


def test_find_matching_words(vectorized_corpus: dtm.VectorizedCorpus):

    vectorized_corpus._token2id = {"bengt": 0, "bertil": 1, "eva": 2, "julia": 3}  # pylint: disable=protected-access

    assert set(vectorized_corpus.find_matching_words(["jens"], 4)) == set()
    assert set(vectorized_corpus.find_matching_words([], 4)) == set()
    assert set(vectorized_corpus.find_matching_words(["bengt"], 4)) == {"bengt"}
    assert set(vectorized_corpus.find_matching_words(["b*"], 4)) == {"bengt", "bertil"}
    assert set(vectorized_corpus.find_matching_words(["|.*a|"], 4)) == {"eva", "julia"}
    assert set(vectorized_corpus.find_matching_words(["*"], 4)) == {"bengt", "bertil", "eva", "julia"}


def test_find_matching_indices(vectorized_corpus: dtm.VectorizedCorpus):

    vectorized_corpus._token2id = {"bengt": 0, "bertil": 1, "eva": 2, "julia": 3}  # pylint: disable=protected-access

    assert set(vectorized_corpus.find_matching_words_indices(["jens"], 4)) == set()
    assert set(vectorized_corpus.find_matching_words_indices([], 4)) == set()
    assert set(vectorized_corpus.find_matching_words_indices(["bengt"], 4)) == {0}
    assert set(vectorized_corpus.find_matching_words_indices(["b*"], 4)) == {0, 1}
    assert set(vectorized_corpus.find_matching_words_indices(["|.*a|"], 4)) == {2, 3}
    assert set(vectorized_corpus.find_matching_words_indices(["*"], 4)) == {0, 1, 2, 3}
