import os

import numpy as np
import pandas as pd
import penelope.corpus.dtm as dtm
import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as corpora
import pytest
import scipy
from penelope.utility.utils import is_strictly_increasing
from sklearn.feature_extraction.text import CountVectorizer
from tests.utils import OUTPUT_FOLDER, create_tokens_reader

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=redefined-outer-name


def flatten(lst):
    return [x for ws in lst for x in ws]


def create_reader() -> readers.TextTokenizer:
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True)
    return reader


def create_corpus() -> corpora.TokenizedCorpus:
    reader = create_reader()
    tokens_transform_opts = corpora.TokensTransformOpts(
        only_any_alphanumeric=True,
        to_lower=True,
        remove_accents=False,
        min_len=2,
        max_len=None,
        keep_numerals=False,
    )
    corpus = corpora.TokenizedCorpus(reader, tokens_transform_opts=tokens_transform_opts)
    return corpus


def create_vectorized_corpus() -> dtm.VectorizedCorpus:
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
    v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return v_corpus


def create_slice_by_n_count_test_corpus() -> dtm.VectorizedCorpus:
    bag_term_matrix = np.array([[1, 1, 4, 1], [0, 2, 3, 0], [0, 3, 2, 0], [0, 4, 1, 3], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    return dtm.VectorizedCorpus(bag_term_matrix, token2id, df)


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


def test_corpus_token_counts(vectorized_corpus):
    assert vectorized_corpus.corpus_token_counts.tolist() == [10, 10, 11, 3]


def test_document_token_counts(vectorized_corpus):
    assert vectorized_corpus.document_token_counts.tolist() == [8, 7, 7, 8, 4]


def test_document_token_counter(vectorized_corpus):
    assert vectorized_corpus.token_counter == {'a': 10, 'b': 10, 'c': 11, 'd': 3}


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
    assert dumped_v_corpus.token_counter == loaded_v_corpus.token_counter
    assert dumped_v_corpus.document_index.to_dict() == loaded_v_corpus.document_index.to_dict()
    assert dumped_v_corpus.token2id == loaded_v_corpus.token2id


def test_load_of_compressed_corpus(text_corpus):

    # Arrange
    dumped_v_corpus: dtm.VectorizedCorpus = dtm.CorpusVectorizer().fit_transform(text_corpus, already_tokenized=True)

    dumped_v_corpus.dump(tag='dump_test', folder=OUTPUT_FOLDER, compressed=True)

    # Act
    loaded_v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus.load(tag='dump_test', folder=OUTPUT_FOLDER)

    # Assert
    assert dumped_v_corpus.token_counter == loaded_v_corpus.token_counter
    assert dumped_v_corpus.document_index.to_dict() == loaded_v_corpus.document_index.to_dict()
    assert dumped_v_corpus.token2id == loaded_v_corpus.token2id


def test_group_by_year_aggregates_bag_term_matrix_to_year_term_matrix(vectorized_corpus):
    v_corpus: dtm.VectorizedCorpus = create_vectorized_corpus()
    c_data = v_corpus.group_by_year()
    expected_ytm = [[4, 3, 7, 1], [6, 7, 4, 2]]
    assert np.allclose(expected_ytm, c_data.bag_term_matrix.todense())


def test_group_by_year_category_aggregates_DTM_to_PTM():

    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2009, 2013, 2014, 2017, 2017]})
    corpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, document_index)

    grouped_corpus = corpus.group_by_period(period='year')
    expected_ytm = [[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [4, 4, 2, 2]]
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())

    grouped_corpus = corpus.group_by_period(period='lustrum')
    expected_ytm = [[2, 1, 4, 1], [4, 5, 5, 0], [4, 4, 2, 2]]
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())

    grouped_corpus = corpus.group_by_period(period='decade')
    expected_ytm = [[2, 1, 4, 1], [8, 9, 7, 2]]
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())

    grouped_corpus = corpus.group_by_period(period='year', fill_gaps=True)
    expected_ytm = np.matrix(
        [
            [2, 1, 4, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 4, 2, 2],
        ]
    )
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())
    assert len(grouped_corpus.document_index) == 9
    assert is_strictly_increasing(grouped_corpus.document_index.index, sort_values=False)


def test_group_by_year_sum_bag_term_matrix_to_year_term_matrix(vectorized_corpus):
    c_data = vectorized_corpus.group_by_year(aggregate='sum', fill_gaps=True)
    expected_ytm = [[4, 3, 7, 1], [6, 7, 4, 2]]
    assert np.allclose(expected_ytm, c_data.bag_term_matrix.todense())
    assert vectorized_corpus.data.dtype == c_data.data.dtype


def test_group_by_year_mean_bag_term_matrix_to_year_term_matrix(vectorized_corpus):
    c_data = vectorized_corpus.group_by_year(aggregate='mean', fill_gaps=True)
    expected_ytm = np.array([np.array([4.0, 3.0, 7.0, 1.0]) / 2.0, np.array([6.0, 7.0, 4.0, 2.0]) / 3.0])
    assert np.allclose(expected_ytm, c_data.bag_term_matrix.todense())


def test_group_by_category_aggregates_bag_term_matrix_to_category_term_matrix(vectorized_corpus):
    """ A more generic version of group_by_year (not used for now) """
    grouped_corpus: dtm.VectorizedCorpus = vectorized_corpus.group_by_category('year')
    expected_ytm = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())


def test_group_by_category_sums_bag_term_matrix_to_category_term_matrix(vectorized_corpus):
    """ A more generic version of group_by_year (not used for now) """
    grouped_corpus = vectorized_corpus.group_by_category(column_name='year')
    expected_ytm = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())


def test_group_by_category_means_bag_term_matrix_to_category_term_matrix(vectorized_corpus):
    """ A more generic version of group_by_year (not used for now) """

    grouped_corpus = vectorized_corpus.group_by_category(column_name='year', aggregate='sum')
    expected_ytm = [np.array([4.0, 3.0, 7.0, 1.0]), np.array([6.0, 7.0, 4.0, 2.0])]
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())

    grouped_corpus = vectorized_corpus.group_by_category(column_name='year', aggregate='mean')
    expected_ytm = np.array([np.array([4.0, 3.0, 7.0, 1.0]) / 2.0, np.array([6.0, 7.0, 4.0, 2.0]) / 3.0])
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())


def test_group_by_year_with_average():

    corpus = [
        "the house had a tiny little mouse",
        "the cat saw the mouse",
        "the mouse ran away from the house",
        "the cat finally ate the mouse",
        "the end of the mouse story",
    ]
    expected_bag_term_matrix = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 2, 0],
            [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0],
        ]
    )

    expected_bag_term_matrix_sums = np.array(
        [expected_bag_term_matrix[[0, 1, 2], :].sum(axis=0), expected_bag_term_matrix[[3, 4], :].sum(axis=0)]
    )

    expected_bag_term_matrix_means = np.array(
        [
            expected_bag_term_matrix[[0, 1, 2], :].sum(axis=0) / 3.0,
            expected_bag_term_matrix[[3, 4], :].sum(axis=0) / 2.0,
        ]
    )

    document_index = pd.DataFrame({'year': [1, 1, 1, 2, 2]})

    vec = CountVectorizer()
    bag_term_matrix = vec.fit_transform(corpus)

    v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(
        bag_term_matrix, token2id=vec.vocabulary_, document_index=document_index
    )

    assert np.allclose(expected_bag_term_matrix, bag_term_matrix.todense())

    y_sum_corpus = v_corpus.group_by_year(aggregate='sum', fill_gaps=True)
    y_mean_corpus = v_corpus.group_by_year(aggregate='mean', fill_gaps=True)

    assert np.allclose(expected_bag_term_matrix_sums, y_sum_corpus.data.todense())
    assert np.allclose(expected_bag_term_matrix_means, y_mean_corpus.data.todense())


def test_normalize_with_default_arguments_returns_matrix_normalized_by_l1_norm_for_each_row():
    bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2014]})
    v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, df)
    n_corpus = v_corpus.normalize()
    E = np.array([[4, 3, 7, 1], [6, 7, 4, 2]]) / (np.array([[15, 19]]).T)
    assert (E == n_corpus.bag_term_matrix).all()


def test_normalize_with_keep_magnitude():
    bag_term_matrix = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    bag_term_matrix = scipy.sparse.csr_matrix(bag_term_matrix)

    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2014]})

    v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, df)
    n_corpus = v_corpus.normalize(keep_magnitude=True)

    factor = 15.0 / 19.0
    E = np.array([[4.0, 3.0, 7.0, 1.0], [6.0 * factor, 7.0 * factor, 4.0 * factor, 2.0 * factor]])
    assert np.allclose(E, n_corpus.bag_term_matrix.todense())


def test_slice_by_n_count_when_exists_tokens_below_count_returns_filtered_corpus(slice_corpus):

    # Act
    t_corpus: dtm.VectorizedCorpus = slice_corpus.slice_by_n_count(6)

    # Assert
    expected_bag_term_matrix = np.array([[2, 1, 4], [2, 2, 3], [2, 3, 2], [2, 4, 1], [2, 0, 1]])

    assert {'a': 0, 'b': 1, 'c': 2} == t_corpus.token2id
    assert {'a': 10, 'b': 10, 'c': 11} == t_corpus.token_counter
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


def test_slice_by_n_count_when_all_below_below_n_count_returns_empty_corpus(slice_corpus):

    t_corpus: dtm.VectorizedCorpus = slice_corpus.slice_by_n_count(20)

    assert {} == t_corpus.token2id
    assert {} == t_corpus.token_counter
    assert (np.empty((5, 0)) == t_corpus.bag_term_matrix).all()


def test_slice_by_n_count_when_all_tokens_above_n_count_returns_same_corpus(slice_corpus):

    t_corpus = slice_corpus.slice_by_n_count(1)

    assert slice_corpus.token2id == t_corpus.token2id
    assert slice_corpus.token_counter == t_corpus.token_counter
    assert np.allclose(slice_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_all_tokens_above_n_count_returns_same_corpus(slice_corpus):

    t_corpus = slice_corpus.slice_by_n_top(4)

    assert slice_corpus.token2id == t_corpus.token2id
    assert slice_corpus.token_counter == t_corpus.token_counter
    assert np.allclose(slice_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_n_top_less_than_n_tokens_returns_corpus_with_top_n_counts(slice_corpus):

    t_corpus = slice_corpus.slice_by_n_top(2)

    expected_bag_term_matrix = np.array([[2, 4], [2, 3], [2, 2], [2, 1], [2, 1]])

    assert {'a': 0, 'c': 1} == t_corpus.token2id
    assert {'a': 10, 'c': 11} == t_corpus.token_counter
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


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


def test_n_top_tokens(vectorized_corpus):
    assert vectorized_corpus.n_top_tokens(2) == {'a': 10, 'c': 11}


def test_stats(vectorized_corpus):
    assert vectorized_corpus.stats() is not None


def test_to_n_top_dataframe(vectorized_corpus):
    assert vectorized_corpus.to_n_top_dataframe(1) is not None


def test_year_range(vectorized_corpus):
    assert vectorized_corpus.year_range() == (2013, 2014)


def test_xs_years(vectorized_corpus):
    assert vectorized_corpus.xs_years().tolist() == [2013, 2014]


def test_token_indices(vectorized_corpus):
    assert vectorized_corpus.token_indices(['a', 'c']) == [0, 2]


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
