import numpy as np
import pandas as pd
import penelope.corpus.dtm as dtm
import pytest
from penelope.co_occurrence import Bundle, to_filename
from penelope.corpus.dtm import VectorizedCorpus
from penelope.corpus.dtm.group import categorize_document_index
from penelope.utility import is_strictly_increasing
from sklearn.feature_extraction.text import CountVectorizer

#  # pylint: disable=redefined-outer-name


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    folder, tag = './tests/test_data/VENUS', 'VENUS'

    filename = to_filename(folder=folder, tag=tag)

    bundle: Bundle = Bundle.load(filename, compute_frame=False)

    return bundle


def test_categorize_document_index(bundle: Bundle):

    corpus: VectorizedCorpus = bundle.corpus
    document_index: pd.DataFrame = bundle.corpus.document_index

    document_index, category_indicies = categorize_document_index(
        corpus.document_index,
        'year',
    )

    assert category_indicies == corpus.document_index.groupby("year").apply(lambda x: x.index.tolist()).to_dict()
    assert document_index is not None


def test_group_corpus_by_document_index(bundle: Bundle):

    corpus: VectorizedCorpus = bundle.corpus.group_by_document_index(period_specifier='year', aggregate='sum')

    assert corpus is not None
    assert corpus.data.shape[0] == len(corpus.document_index)
    assert len(corpus.document_index) == 5
    assert set(corpus.document_index.category.tolist()) == set([1945, 1958, 1978, 1997, 2017])

    corpus: VectorizedCorpus = bundle.corpus.group_by_document_index(period_specifier='lustrum', aggregate='sum')
    assert corpus is not None
    assert corpus.data.shape[0] == len(corpus.document_index)
    assert len(corpus.document_index) == 5
    assert set(corpus.document_index.category.tolist()) == set([1945, 1955, 1975, 1995, 2015])

    corpus: VectorizedCorpus = bundle.corpus.group_by_document_index(period_specifier='decade', aggregate='sum')
    assert corpus is not None
    assert corpus.data.shape[0] == len(corpus.document_index)
    assert len(corpus.document_index) == 5
    assert set(corpus.document_index.category.tolist()) == set([1940, 1950, 1970, 1990, 2010])


def test_group_by_year_aggregates_bag_term_matrix_to_year_term_matrix(vectorized_corpus):
    c_data = vectorized_corpus.group_by_year()
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
