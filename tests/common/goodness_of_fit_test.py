import numpy as np
import pandas as pd
import pytest
import scipy

import penelope.common.distance_metrics as distance_metrics
import penelope.common.goodness_of_fit as gof
import penelope.utility as utility
from penelope.corpus import VectorizedCorpus

logger = utility.get_logger()


def create_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus = VectorizedCorpus(bag_term_matrix, token2id=token2id, document_index=df)
    return v_corpus


def test_ols_when_x_equals_y_k_equals_one():

    xs = [1, 2, 3, 4]
    ys = [1, 2, 3, 4]
    m, k, _, _, _ = distance_metrics.fit_ordinary_least_square(ys, xs)

    assert np.allclose(0.0, m)
    assert 1.0 == pytest.approx(k)


def test_ols_when_x_equals_y_k_equals_expected():
    # y = 3 * x + 4
    #
    xs = np.array([1, 2, 3, 4])
    ys = [7, 10, 13, 16]
    m, k, _, _, _ = distance_metrics.fit_ordinary_least_square(ys, xs)

    assert np.allclose(4.0, m)
    assert 3.0 == pytest.approx(k)


def test_gof_by_l2_norm():

    m = np.array(
        [
            [0.10, 0.11, 0.10, 0.09, 0.09, 0.11, 0.10, 0.10, 0.12, 0.08],
            [0.10, 0.10, 0.10, 0.08, 0.12, 0.12, 0.09, 0.09, 0.12, 0.08],
            [0.03, 0.02, 0.61, 0.02, 0.03, 0.07, 0.06, 0.05, 0.06, 0.05],
        ]
    )

    # The following will yield 0.0028, 0.0051, and 0.4529 for the rows:

    expected = [0.0028, 0.0051, 0.4529]

    result = gof.gof_by_l2_norm(m, axis=1)

    assert np.allclose(expected, result.round(4))


def test_fit_ordinary_least_square():

    Y = [1, 3, 4, 5, 2, 3, 4]
    X = list(range(1, 8))
    m, k, _, _, _ = distance_metrics.fit_ordinary_least_square(Y, X)
    assert round(k, 6) == round(0.25, 6)
    assert round(m, 6) == round(2.14285714, 6)


def test_fit_ordinary_least_square_to_horizontal_line():
    Y = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    X = list(range(1, 8))
    m, k, _, _, _ = distance_metrics.fit_ordinary_least_square(Y, X)
    assert round(k, 6) == round(0.0, 6)
    assert round(m, 6) == round(2.0, 6)


def test_fit_ordinary_least_square_to_3_14_x_plus_4():

    kp = 3.14
    mp = 4.0

    X = list(range(1, 8))
    Y = [kp * x + mp for x in range(1, 8)]

    m, k, _, _, _ = distance_metrics.fit_ordinary_least_square(Y, X)

    assert round(kp, 6) == round(k, 6)
    assert round(mp, 6) == round(m, 6)


def test_compute_goddness_of_fits_to_uniform():

    expected_columns = [
        'token',
        'word_count',
        'l2_norm',
        'slope',
        'chi2_stats',
        'earth_mover',
        'entropy',
        'kld',
        'skew',
    ]
    corpus = create_vectorized_corpus()

    df_gof = gof.compute_goddness_of_fits_to_uniform(corpus=corpus)

    assert df_gof is not None
    assert expected_columns == list(df_gof.columns)


def test_compute_goddness_of_fits_to_uniform_with_reduced_columns():

    metrics = ['l2_norm', 'slope']
    expected_columns = ['token', 'word_count'] + metrics
    corpus = create_vectorized_corpus()

    df_gof = gof.compute_goddness_of_fits_to_uniform(corpus=corpus, metrics=metrics)

    assert df_gof is not None
    assert expected_columns == list(df_gof.columns)


def test_distance_metrics_fit_polynomial():

    expected_values = np.array(
        [
            [2.00000000e00, 2.00000000e00, 3.80000000e00, 4.00000000e-01],
            [-1.93554769e-16, -3.13963176e-17, -8.00000000e-01, 1.00000000e-01],
        ]
    )
    corpus = create_vectorized_corpus()
    dtm = corpus.data
    ys = dtm.todense()
    xs = range(0, ys.shape[0])

    fitted_values = distance_metrics.fit_polynomial(xs=xs, ys=ys, deg=1)

    assert np.allclose(expected_values, fitted_values)


def test_gof_by_polynomial():

    expected_values = np.array(
        [
            [2.00000000e00, 2.00000000e00, 3.80000000e00, 4.00000000e-01],
            [-1.93554769e-16, -3.13963176e-17, -8.00000000e-01, 1.00000000e-01],
        ]
    )
    corpus = create_vectorized_corpus()

    df = gof.get_gof_by_polynomial(corpus.data)

    assert np.allclose(expected_values.T, df[['intercept', 'slope']].to_numpy())


def test_gof_get_skew():

    row = np.arange(0, 8)
    col = np.zeros(8)
    data = np.array([2, 8, 0, 4, 1, 9, 9, 0])

    dtm = scipy.sparse.csr_matrix((data, (row, col)), shape=(8, 1))

    df = gof.get_skew(dtm)
    assert df['skew'].iloc[0] == 0.2650554122698573

    # skew([1, 2, 3, 4, 5])
    # 0.0
    # skew([2, 8, 0, 4, 1, 9, 9, 0])
    # 0.2650554122698573


def create_single_column_sparse_csr_matrix(data):
    row = np.arange(0, len(data))
    col = np.zeros(len(data))
    dtm = scipy.sparse.csr_matrix((data, (row, col)), shape=(len(data), 1))
    return dtm


def test_get_gof_by_polynomial():

    data = [2, 8, 0, 4, 1, 9, 9, 0]
    expected_result = np.polyfit(np.arange(0, 8), data, 1)
    dtm = create_single_column_sparse_csr_matrix(data)

    df = gof.get_gof_by_polynomial(dtm)

    assert np.allclose([expected_result[0]], df['slope'])
    assert np.allclose([expected_result[1]], df['intercept'])


def test_get_earth_mover_distance():

    data = [2, 8, 0, 4, 1, 9, 9, 0]
    expected_result = distance_metrics.earth_mover_distance(np.array(data))
    dtm = create_single_column_sparse_csr_matrix(data)

    df = gof.get_earth_mover_distance(dtm)

    assert np.allclose([expected_result], df['earth_mover'])


def test_get_entropy_to_uniform():

    data = [2, 8, 0, 4, 1, 9, 9, 0]
    expected_result = distance_metrics.entropy(np.array(data))
    dtm = create_single_column_sparse_csr_matrix(data)

    df = gof.get_entropy_to_uniform(dtm)

    assert np.allclose([expected_result], df['entropy'])


def test_get_kullback_leibler_divergence_to_uniform():

    data = [2, 8, 0, 4, 1, 9, 9, 0]
    expected_result = distance_metrics.kullback_leibler_divergence_to_uniform(np.array(data))
    dtm = create_single_column_sparse_csr_matrix(data)

    df = gof.get_kullback_leibler_divergence_to_uniform(dtm)

    assert np.allclose([expected_result], df['kld'])


def test_get_gof_chisquare_to_uniform():

    data = [2, 8, 0, 4, 1, 9, 9, 0]
    expected_result = distance_metrics.gof_chisquare_to_uniform(np.array(data))
    dtm = create_single_column_sparse_csr_matrix(data)

    df = gof.get_gof_chisquare_to_uniform(dtm)

    assert np.allclose(expected_result[0], df['chi2_stats'])
    assert np.allclose(expected_result[1], df['chi2_p'])
