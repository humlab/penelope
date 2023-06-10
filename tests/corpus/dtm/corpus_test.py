import os

import numpy as np
import pandas as pd
import pytest

from penelope.corpus import (
    CorpusVectorizer,
    TokenizedCorpus,
    TokensTransformOpts,
    VectorizedCorpus,
    find_matching_words_in_vocabulary,
)
from tests.utils import OUTPUT_FOLDER, create_abc_corpus, create_tokens_reader, create_vectorized_corpus

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=redefined-outer-name


@pytest.fixture
def text_corpus() -> TokenizedCorpus:
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, text_transforms="normalize-whitespace,dehyphen")
    transform_opts = TokensTransformOpts(
        transforms={
            'only-any-alphanumeric': True,
            'to-lower': True,
            'min-len': 2,
            'remove-numerals': True,
        }
    )
    corpus = TokenizedCorpus(reader, transform_opts=transform_opts)
    return corpus


@pytest.fixture
def corpus() -> VectorizedCorpus:
    return create_vectorized_corpus()


@pytest.fixture
def slice_corpus() -> VectorizedCorpus:
    return create_vectorized_corpus()


def test_vocabulary(corpus: VectorizedCorpus):
    assert corpus.vocabulary == ['a', 'b', 'c', 'd']


def test_term_frequencies(corpus: VectorizedCorpus):
    assert corpus.term_frequency.tolist() == [10, 10, 11, 3]


def test_document_token_counts(corpus: VectorizedCorpus):
    assert corpus.document_token_counts.tolist() == [8, 7, 7, 8, 4]


def test_n_terms(corpus: VectorizedCorpus):
    assert corpus.n_tokens == 4


def test_n_docs(corpus: VectorizedCorpus):
    assert corpus.n_docs == 5


def test_bag_term_matrix_to_bag_term_docs(corpus: VectorizedCorpus):
    doc_ids = (
        0,
        1,
    )
    expected = [['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'], ['a', 'a', 'b', 'b', 'c', 'c', 'c']]
    docs = corpus.to_bag_of_terms(doc_ids)
    assert expected == ([list(d) for d in docs])

    expected = [
        ['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'],
        ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'd'],
        ['a', 'a', 'c', 'd'],
    ]
    docs = corpus.to_bag_of_terms()
    assert expected == ([list(d) for d in docs])


def test_load_of_uncompressed_corpus(text_corpus):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Arrange
    corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)

    corpus.dump(tag='dump_test', folder=OUTPUT_FOLDER, compressed=False)

    # Act
    loaded_corpus: VectorizedCorpus = VectorizedCorpus.load(tag='dump_test', folder=OUTPUT_FOLDER)

    # Assert
    assert (corpus.term_frequency == loaded_corpus.term_frequency).all()
    assert corpus.document_index.to_dict() == loaded_corpus.document_index.to_dict()
    assert corpus.token2id == loaded_corpus.token2id


def test_load_of_compressed_corpus(text_corpus: TokenizedCorpus):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Arrange
    dumped_corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)

    dumped_corpus.dump(tag='dump_test', folder=OUTPUT_FOLDER, compressed=True)

    # Act
    loaded_corpus: VectorizedCorpus = VectorizedCorpus.load(tag='dump_test', folder=OUTPUT_FOLDER)

    # Assert
    assert (dumped_corpus.term_frequency == loaded_corpus.term_frequency).all()
    assert dumped_corpus.document_index.to_dict() == loaded_corpus.document_index.to_dict()
    assert dumped_corpus.token2id == loaded_corpus.token2id


def test_id2token_is_reversed_token2id(corpus: VectorizedCorpus):
    id2token = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    assert id2token == corpus.id2token


def test_normalize_by_raw_counts():
    corpus: VectorizedCorpus = VectorizedCorpus(
        bag_term_matrix=np.array([[4, 3, 7, 1], [6, 7, 4, 2]]),
        token2id={'a': 0, 'b': 1, 'c': 2, 'd': 3},
        document_index=pd.DataFrame({'year': [2013, 2014]}),
    )

    n_corpus = corpus.normalize()
    t_corpus = corpus.normalize_by_raw_counts()
    assert np.allclose(t_corpus.data.todense(), n_corpus.data.todense())


def test_dump_and_store_of_corpus_with_empty_trailing_row() -> VectorizedCorpus:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [0, 0, 0, 0]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014]})
    corpus: VectorizedCorpus = VectorizedCorpus(bag_term_matrix, token2id=token2id, document_index=document_index)

    corpus.dump(tag="ZERO", folder="./tests/output")

    loaded_corpus = VectorizedCorpus.load(tag="ZERO", folder="./tests/output")

    assert corpus.data.shape == loaded_corpus.data.shape


def test_find_matching_words_in_vocabulary():
    token2id = {"bengt": 0, "bertil": 1, "eva": 2, "julia": 3}

    assert find_matching_words_in_vocabulary(token2id, ["jens"]) == set()
    assert find_matching_words_in_vocabulary(token2id, []) == set()
    assert find_matching_words_in_vocabulary(token2id, ["bengt"]) == {"bengt"}
    assert find_matching_words_in_vocabulary(token2id, ["b*"]) == {"bengt", "bertil"}
    assert find_matching_words_in_vocabulary(token2id, ["|.*a|"]) == {"eva", "julia"}
    assert find_matching_words_in_vocabulary(token2id, ["*"]) == {"bengt", "bertil", "eva", "julia"}


def test_to_dense(corpus: VectorizedCorpus):
    assert corpus.todense() is not None


def test_get_word_vector(corpus: VectorizedCorpus):
    assert corpus.get_word_vector('b').tolist() == [1, 2, 3, 4, 0]


def test_filter():
    corpus: VectorizedCorpus = create_abc_corpus(
        dtm=[[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]],
        document_years=[2012, 2013, 2014, 2014, 2015],
    )

    filtered_corpus: VectorizedCorpus = corpus.filter(lambda x: x['year'] == 2014)
    assert filtered_corpus.shape == (2, 4)
    assert len(filtered_corpus.document_index) == 2
    assert len(filtered_corpus.document_index) == 2


def test_pick_top_tf_map(corpus: VectorizedCorpus):
    assert corpus.pick_top_tf_map(2) == {'a': 10, 'c': 11}


def test_stats(corpus: VectorizedCorpus):
    assert corpus.stats() is not None


def test_to_n_top_dataframe(corpus: VectorizedCorpus):
    assert corpus.to_n_top_dataframe(1) is not None


def test_token_indices(corpus: VectorizedCorpus):
    assert corpus.token_indices(['a', 'c', 'z']) == [0, 2]


def test_tf_idf(corpus: VectorizedCorpus):
    assert corpus.tf_idf() is not None


def test_to_bag_of_terms(corpus: VectorizedCorpus):
    expected_docs = [
        ['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'],
        ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'd'],
        ['a', 'a', 'c', 'd'],
    ]
    assert [list(x) for x in corpus.to_bag_of_terms()] == expected_docs


def test_get_top_n_words(corpus: VectorizedCorpus):
    assert corpus.get_top_n_words(n=2) == [('c', 11), ('a', 10)]


def test_co_occurrence_matrix(corpus: VectorizedCorpus):
    m = corpus.co_occurrence_matrix()
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


def test_find_matching_words(corpus: VectorizedCorpus):
    corpus._token2id = {"bengt": 0, "bertil": 1, "eva": 2, "julia": 3}  # pylint: disable=protected-access

    assert set(corpus.find_matching_words(["jens"], 4)) == set()
    assert set(corpus.find_matching_words([], 4)) == set()
    assert set(corpus.find_matching_words(["bengt"], 4)) == {"bengt"}
    assert set(corpus.find_matching_words(["b*"], 4)) == {"bengt", "bertil"}
    assert set(corpus.find_matching_words(["|.*a|"], 4)) == {"eva", "julia"}
    assert set(corpus.find_matching_words(["*"], 4)) == {"bengt", "bertil", "eva", "julia"}


def test_find_matching_indices(corpus: VectorizedCorpus):
    corpus._token2id = {"bengt": 0, "bertil": 1, "eva": 2, "julia": 3}  # pylint: disable=protected-access

    assert set(corpus.find_matching_words_indices(["jens"], 4)) == set()
    assert set(corpus.find_matching_words_indices([], 4)) == set()
    assert set(corpus.find_matching_words_indices(["bengt"], 4)) == {0}
    assert set(corpus.find_matching_words_indices(["b*"], 4)) == {0, 1}
    assert set(corpus.find_matching_words_indices(["|.*a|"], 4)) == {2, 3}
    assert set(corpus.find_matching_words_indices(["*"], 4)) == {0, 1, 2, 3}
