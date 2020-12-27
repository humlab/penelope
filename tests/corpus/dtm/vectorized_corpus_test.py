import os
from token import tok_name

import numpy as np
import pandas as pd
import penelope.corpus.dtm as dtm
import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as corpora
import scipy
from penelope.utility.utils import is_strictly_increasing
from sklearn.feature_extraction.text import CountVectorizer
from tests.utils import OUTPUT_FOLDER, create_tokens_reader


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
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return v_corpus


def create_slice_by_n_count_test_corpus() -> dtm.VectorizedCorpus:
    bag_term_matrix = np.array([[1, 1, 4, 1], [0, 2, 3, 0], [0, 3, 2, 0], [0, 4, 1, 3], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    return dtm.VectorizedCorpus(bag_term_matrix, token2id, df)


os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def test_bag_term_matrix_to_bag_term_docs():

    v_corpus: dtm.VectorizedCorpus = create_vectorized_corpus()

    doc_ids = (
        0,
        1,
    )
    expected = [['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'], ['a', 'a', 'b', 'b', 'c', 'c', 'c']]
    docs = v_corpus.to_bag_of_terms(doc_ids)
    assert expected == ([list(d) for d in docs])

    expected = [
        ['a', 'a', 'b', 'c', 'c', 'c', 'c', 'd'],
        ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'c', 'c'],
        ['a', 'a', 'b', 'b', 'b', 'b', 'c', 'd'],
        ['a', 'a', 'c', 'd'],
    ]
    docs = v_corpus.to_bag_of_terms()
    assert expected == ([list(d) for d in docs])


def test_load_of_uncompressed_corpus():

    # Arrange
    corpus = create_corpus()
    dumped_v_corpus: dtm.VectorizedCorpus = dtm.CorpusVectorizer().fit_transform(corpus, already_tokenized=True)

    dumped_v_corpus.dump(tag='dump_test', folder=OUTPUT_FOLDER, compressed=False)

    # Act
    loaded_v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus.load(tag='dump_test', folder=OUTPUT_FOLDER)

    # Assert
    assert dumped_v_corpus.token_counter == loaded_v_corpus.token_counter
    assert dumped_v_corpus.document_index.to_dict() == loaded_v_corpus.document_index.to_dict()
    assert dumped_v_corpus.token2id == loaded_v_corpus.token2id


def test_load_of_compressed_corpus():

    # Arrange
    corpus = create_corpus()
    dumped_v_corpus: dtm.VectorizedCorpus = dtm.CorpusVectorizer().fit_transform(corpus, already_tokenized=True)

    dumped_v_corpus.dump(tag='dump_test', folder=OUTPUT_FOLDER, compressed=True)

    # Act
    loaded_v_corpus: dtm.VectorizedCorpus = dtm.VectorizedCorpus.load(tag='dump_test', folder=OUTPUT_FOLDER)

    # Assert
    assert dumped_v_corpus.token_counter == loaded_v_corpus.token_counter
    assert dumped_v_corpus.document_index.to_dict() == loaded_v_corpus.document_index.to_dict()
    assert dumped_v_corpus.token2id == loaded_v_corpus.token2id


def test_group_by_year_aggregates_bag_term_matrix_to_year_term_matrix():
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


def test_group_by_year_sum_bag_term_matrix_to_year_term_matrix():
    v_corpus: dtm.VectorizedCorpus = create_vectorized_corpus()
    c_data = v_corpus.group_by_year(aggregate='sum', fill_gaps=True)
    expected_ytm = [[4, 3, 7, 1], [6, 7, 4, 2]]
    assert np.allclose(expected_ytm, c_data.bag_term_matrix.todense())
    assert v_corpus.data.dtype == c_data.data.dtype


def test_group_by_year_mean_bag_term_matrix_to_year_term_matrix():
    v_corpus: dtm.VectorizedCorpus = create_vectorized_corpus()
    c_data = v_corpus.group_by_year(aggregate='mean', fill_gaps=True)
    expected_ytm = np.array([np.array([4.0, 3.0, 7.0, 1.0]) / 2.0, np.array([6.0, 7.0, 4.0, 2.0]) / 3.0])
    assert np.allclose(expected_ytm, c_data.bag_term_matrix.todense())


def test_group_by_category_aggregates_bag_term_matrix_to_category_term_matrix():
    """ A more generic version of group_by_year (not used for now) """
    corpus: dtm.VectorizedCorpus = create_vectorized_corpus()
    grouped_corpus: dtm.VectorizedCorpus = corpus.group_by_category('year')
    expected_ytm = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())


def test_group_by_category_sums_bag_term_matrix_to_category_term_matrix():
    """ A more generic version of group_by_year (not used for now) """
    corpus = create_vectorized_corpus()
    grouped_corpus = corpus.group_by_category(column_name='year')
    expected_ytm = np.array([[4, 3, 7, 1], [6, 7, 4, 2]])
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())


def test_group_by_category_means_bag_term_matrix_to_category_term_matrix():
    """ A more generic version of group_by_year (not used for now) """
    corpus = create_vectorized_corpus()

    grouped_corpus = corpus.group_by_category(column_name='year', aggregate='sum')
    expected_ytm = [np.array([4.0, 3.0, 7.0, 1.0]), np.array([6.0, 7.0, 4.0, 2.0])]
    assert np.allclose(expected_ytm, grouped_corpus.bag_term_matrix.todense())

    grouped_corpus = corpus.group_by_category(column_name='year', aggregate='mean')
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


def test_slice_by_n_count_when_exists_tokens_below_count_returns_filtered_corpus():

    v_corpus: dtm.VectorizedCorpus = create_slice_by_n_count_test_corpus()

    # Act
    t_corpus: dtm.VectorizedCorpus = v_corpus.slice_by_n_count(6)

    # Assert
    expected_bag_term_matrix = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [0, 1]])

    assert {'b': 0, 'c': 1} == t_corpus.token2id
    assert {'b': 10, 'c': 11} == t_corpus.token_counter
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


def test_slice_by_n_count_when_all_below_below_n_count_returns_empty_corpus():
    v_corpus: dtm.VectorizedCorpus = create_slice_by_n_count_test_corpus()

    t_corpus: dtm.VectorizedCorpus = v_corpus.slice_by_n_count(20)

    assert {} == t_corpus.token2id
    assert {} == t_corpus.token_counter
    assert (np.empty((5, 0)) == t_corpus.bag_term_matrix).all()


def test_slice_by_n_count_when_all_tokens_above_n_count_returns_same_corpus():
    v_corpus: dtm.VectorizedCorpus = create_slice_by_n_count_test_corpus()

    t_corpus = v_corpus.slice_by_n_count(1)

    assert v_corpus.token2id == t_corpus.token2id
    assert v_corpus.token_counter == t_corpus.token_counter
    assert np.allclose(v_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_all_tokens_above_n_count_returns_same_corpus():
    v_corpus: dtm.VectorizedCorpus = create_slice_by_n_count_test_corpus()

    t_corpus = v_corpus.slice_by_n_top(4)

    assert v_corpus.token2id == t_corpus.token2id
    assert v_corpus.token_counter == t_corpus.token_counter
    assert np.allclose(v_corpus.bag_term_matrix.todense().A, t_corpus.bag_term_matrix.todense().A)


def test_slice_by_n_top_when_n_top_less_than_n_tokens_returns_corpus_with_top_n_counts():
    v_corpus: dtm.VectorizedCorpus = create_slice_by_n_count_test_corpus()

    t_corpus = v_corpus.slice_by_n_top(2)

    expected_bag_term_matrix = np.array([[1, 4], [2, 3], [3, 2], [4, 1], [0, 1]])

    assert {'b': 0, 'c': 1} == t_corpus.token2id
    assert {'b': 10, 'c': 11} == t_corpus.token_counter
    assert (expected_bag_term_matrix == t_corpus.bag_term_matrix).all()


def test_id2token_is_reversed_token2id():
    corpus = create_vectorized_corpus()
    id2token = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
    assert id2token == corpus.id2token


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
