from __future__ import annotations

from typing import List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from penelope import utility as pu
from penelope.common.keyness import KeynessMetric
from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends import TrendsComputeOpts, TrendsData
from tests.fixtures import simple_corpus_with_pivot_keys


def test_TrendsData_create():
    pass


def test_TrendsData_update():

    data = TrendsData(corpus=simple_corpus_with_pivot_keys(), n_top=10)

    assert isinstance(data.gof_data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.gof_data.most_deviating_overview, pd.DataFrame)
    assert isinstance(data.gof_data.most_deviating, pd.DataFrame)


def test_TrendsData_remember():
    pass


@pytest.mark.parametrize(
    'temporal_key,normalize,fill_gaps,expected_tf_sums,expected_shape',
    [
        ('year', False, True, [8, 0, 0, 0, 7, 7, 0, 0, 12], (9, 4)),
        ('year', False, False, [8, 7, 7, 12], (4, 4)),
        ('year', True, True, [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], (9, 4)),
        ('year', True, False, [1.0, 1.0, 1.0, 1.0], (4, 4)),
        ('decade', True, False, [1.0, 1.0], (2, 4)),
        ('decade', False, False, [8, 26], (2, 4)),
        ('lustrum', False, False, [8, 14, 12], (3, 4)),
        ('lustrum', True, False, [1.0, 1.0, 1.0], (3, 4)),
    ],
)
def test_trends_data_transform_normalize_fill_gaps_without_pivot_keys(
    temporal_key: str,
    normalize: bool,
    fill_gaps: bool,
    expected_tf_sums: List[int | float],
    expected_shape: Tuple[int, int],
):

    trends_data: TrendsData = TrendsData(corpus=simple_corpus_with_pivot_keys(), n_top=100)

    corpus: VectorizedCorpus = trends_data.transform(
        TrendsComputeOpts(normalize=normalize, keyness=KeynessMetric.TF, temporal_key=temporal_key, fill_gaps=fill_gaps)
    ).transformed_corpus

    expected_columns = {temporal_key, 'filename', 'document_name', 'n_raw_tokens', 'year', 'n_documents', 'document_id'}
    assert set(corpus.document_index.columns).intersection(expected_columns) == expected_columns

    assert corpus.data.shape == expected_shape
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array(expected_tf_sums))
    assert temporal_key in corpus.document_index.columns


Opts = pu.PropertyValueMaskingOpts


@pytest.mark.parametrize(
    'temporal_key,pivot_keys,pivot_keys_filter,normalize,fill_gaps,expected_tf_sums,expected_shape',
    [
        ('year', ['color_id'], Opts(), False, True, [8, 0, 0, 0, 7, 7, 0, 0, 8, 4], (10, 4)),
        ('year', ['color_id'], Opts(), False, False, [8, 7, 7, 8, 4], (5, 4)),
        ('year', ['color_id'], Opts(color_id=0), False, False, [8, 7], (2, 4)),
        ('year', ['color_id'], Opts(color_id=3), False, False, [4], (1, 4)),
        ('year', ['color_id', 'cov_id'], Opts(), False, False, [8, 7, 7, 8, 4], (5, 4)),
        ('year', ['color_id', 'cov_id'], Opts(cov_id=2), False, False, [7, 8], (2, 4)),
        ('decade', ['color_id'], Opts(), False, True, [8, 7, 7, 8, 4], (5, 4)),
        ('decade', ['color_id'], Opts(), False, False, [8, 7, 7, 8, 4], (5, 4)),
    ],
)
def test_trends_data_transform_normalize_fill_gaps_with_pivot_keys(
    temporal_key: str,
    pivot_keys: List[str],
    pivot_keys_filter: pu.PropertyValueMaskingOpts,
    normalize: bool,
    fill_gaps: bool,
    expected_tf_sums: List[int | float],
    expected_shape: Tuple[int, int],
):
    trends_data: TrendsData = TrendsData(corpus=simple_corpus_with_pivot_keys(), n_top=100)

    corpus: VectorizedCorpus = trends_data.transform(
        TrendsComputeOpts(
            normalize=normalize,
            keyness=KeynessMetric.TF,
            temporal_key=temporal_key,
            pivot_keys_id_names=pivot_keys,
            pivot_keys_filter=pivot_keys_filter,
            fill_gaps=fill_gaps,
        )
    ).transformed_corpus

    assert corpus.data.shape == expected_shape
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array(expected_tf_sums))
    assert temporal_key in corpus.document_index.columns
    assert set(corpus.document_index.columns).intersection(set(pivot_keys)) == set(pivot_keys)


# def test_trends_data_tf_idf():
#     trends_data: TrendsData = TrendsData().update(
#         corpus=simple_corpus_with_pivot_keys(), corpus_folder='./tests/test_data', corpus_tag="dummy", n_top=100
#     )
#     corpus: VectorizedCorpus = trends_data.get_corpus(TrendsComputeOpts(normalize=True, keyness=KeynessMetric.TF_IDF, time_period='year'))
#     assert corpus.data.shape == (9, 4)
#     # assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 1.0]))


def test_trends_data_top_terms():
    temporal_key: str = 'year'
    trends_data: TrendsData = TrendsData(corpus=simple_corpus_with_pivot_keys(), n_top=100)
    corpus = trends_data.transform(
        TrendsComputeOpts(normalize=False, keyness=KeynessMetric.TF, temporal_key='year')
    ).transformed_corpus
    assert temporal_key in corpus.document_index
    assert 'time_period' in corpus.document_index

    n_top = 4
    df = corpus.get_top_terms(category_column='year', n_top=n_top, kind='token')
    assert df is not None
    assert df.columns.tolist() == ['2009', '2013', '2014', '2017']
    assert df['2009'].tolist() == ['c', 'a', 'b', 'd']
    assert df['2013'].tolist() == ['c', 'a', 'b', '*']

    df = corpus.get_top_terms(category_column='year', n_top=n_top, kind='token/count')
    assert df is not None
    assert df.columns.tolist() == ['2009', '2013', '2014', '2017']
    assert df['2009'].tolist() == ['c/4', 'a/2', 'b/1', 'd/1']
    assert df['2013'].tolist() == ['c/3', 'a/2', 'b/2', '*/0']

    df = corpus.get_top_terms(category_column='year', n_top=n_top, kind='token+count')
    assert df is not None
    assert df.columns.tolist() == [
        '2009',
        '2009/Count',
        '2013',
        '2013/Count',
        '2014',
        '2014/Count',
        '2017',
        '2017/Count',
    ]
    assert df['2009'].tolist() == ['c', 'a', 'b', 'd']
    assert df['2009/Count'].tolist() == [4, 2, 1, 1]

    corpus = trends_data.transform(
        TrendsComputeOpts(normalize=False, keyness=KeynessMetric.TF, temporal_key='lustrum')
    ).transformed_corpus
    df = corpus.get_top_terms(category_column='lustrum', n_top=n_top, kind='token')
    assert df is not None
    assert df.columns.tolist() == ['2005', '2010', '2015']
    assert df['2005'].tolist() == ['c', 'a', 'b', 'd']
    assert df['2010'].tolist() == ['b', 'c', 'a', '*']

    corpus = trends_data.transform(
        TrendsComputeOpts(normalize=False, keyness=KeynessMetric.TF, temporal_key='decade')
    ).transformed_corpus
    df = corpus.get_top_terms(category_column='decade', n_top=n_top, kind='token')
    assert df is not None
    assert df.columns.tolist() == ['2000', '2010']
    assert df['2000'].tolist() == ['c', 'a', 'b', 'd']
    assert df['2010'].tolist() == ['b', 'a', 'c', 'd']

    corpus = trends_data.transform(
        TrendsComputeOpts(normalize=True, keyness=KeynessMetric.TF, temporal_key='decade')
    ).transformed_corpus
    df = corpus.get_top_terms(category_column='decade', n_top=n_top, kind='token+count')
    assert df is not None
    assert df.columns.tolist() == ['2000', '2000/Count', '2010', '2010/Count']
    assert np.allclose(
        df['2010/Count'].tolist(), [0.34615384615384615, 0.3076923076923077, 0.2692307692307693, 0.07692307692307693]
    )

    corpus = trends_data.transform(
        TrendsComputeOpts(normalize=True, keyness=KeynessMetric.TF_IDF, temporal_key='decade')
    ).transformed_corpus
    df = corpus.get_top_terms(category_column='decade', n_top=n_top, kind='token+count')
    assert df is not None
    assert df.columns.tolist() == ['2000', '2000/Count', '2010', '2010/Count']
    assert np.allclose(
        df['2010/Count'].tolist(), [0.08474405764438495, 0.07812263452765607, 0.06537949635506012, 0.030446904614286802]
    )


@patch('penelope.common.goodness_of_fit.compute_goddness_of_fits_to_uniform', lambda *_, **__: Mock(spec=pd.DataFrame))
@patch('penelope.common.goodness_of_fit.compile_most_deviating_words', lambda *_, **__: Mock(spec=pd.DataFrame))
@patch('penelope.common.goodness_of_fit.get_most_deviating_words', lambda *_, **__: Mock(spec=pd.DataFrame))
def test_group_by_year():

    corpus: VectorizedCorpus = Mock(spec=VectorizedCorpus)
    corpus = corpus.group_by_year()

    trends_data: TrendsData = TrendsData(corpus=corpus, n_top=100)

    assert trends_data is not None


def test_find_word_indices():
    trends_data = TrendsData(corpus=simple_corpus_with_pivot_keys())
    indices = trends_data.find_word_indices(
        TrendsComputeOpts(
            temporal_key='year', normalize=False, smooth=False, keyness=KeynessMetric.TF, words=["c"], top_count=2
        )
    )
    assert indices == [2]


def test_find_words():
    pass
