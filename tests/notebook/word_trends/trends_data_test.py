from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from penelope.co_occurrence import KeynessMetric
from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends import TrendsData, TrendsOpts
from tests.utils import OUTPUT_FOLDER


def simple_corpus():
    corpus = VectorizedCorpus(
        bag_term_matrix=np.array(
            [
                [2, 1, 4, 1],
                [2, 2, 3, 0],
                [2, 3, 2, 0],
                [2, 4, 1, 1],
                [2, 0, 1, 1],
            ]
        ),
        token2id={'a': 0, 'b': 1, 'c': 2, 'd': 3},
        document_index=pd.DataFrame(
            {
                'year': [2009, 2013, 2014, 2017, 2017],
                'document_id': [0, 1, 2, 3, 4],
                'document_name': [f'doc_{y}_{i}' for i, y in enumerate(range(0, 5))],
                'filename': [f'doc_{y}_{i}.txt' for i, y in enumerate(range(0, 5))],
            }
        ),
    )
    return corpus


def test_TrendsData_create():
    pass


def test_TrendsData_update():

    data = TrendsData().update(
        corpus=simple_corpus(),
        corpus_folder=OUTPUT_FOLDER,
        corpus_tag="dummy",
        n_count=10,
    )

    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating_overview, pd.DataFrame)
    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating, pd.DataFrame)


def test_TrendsData_remember():
    pass


def test_TrendsData_get_corpus():
    expected_category_column: str = 'time_period'
    trends_data: TrendsData = TrendsData().update(
        corpus=simple_corpus(), corpus_folder='./tests/test_data', corpus_tag="dummy", n_count=100
    )

    corpus: VectorizedCorpus = trends_data.get_corpus(
        TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='year')
    )
    assert corpus.data.shape == (9, 4)  # Shape of 'year' should include years without documents (gaps are filled)
    assert corpus.data.shape == trends_data.corpus.data.shape
    assert corpus.data.sum() == trends_data.corpus.data.sum()
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([8.0, 0.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0, 12.0]))
    assert 'year' in corpus.document_index.columns
    assert expected_category_column in corpus.document_index.columns

    corpus: VectorizedCorpus = trends_data.get_corpus(
        TrendsOpts(normalize=True, keyness=KeynessMetric.TF, group_by='year')
    )
    assert corpus.data.shape == (9, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]))
    assert 'year' in corpus.document_index.columns
    assert expected_category_column in corpus.document_index.columns

    expected_columns = [
        expected_category_column,
        'filename',
        'document_name',
        'n_docs_size',
        'n_raw_tokens_sum',
        'year_min',
        'year_max',
        'year_size',
        'year',
        'document_id',
        'n_raw_tokens',
    ]

    corpus: VectorizedCorpus = trends_data.get_corpus(
        TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='lustrum')
    )
    assert corpus.data.shape == (3, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([8.0, 14.0, 12.0]))
    assert (corpus.document_index.columns == expected_columns).all()

    corpus: VectorizedCorpus = trends_data.get_corpus(
        TrendsOpts(normalize=True, keyness=KeynessMetric.TF, group_by='lustrum')
    )
    assert corpus.data.shape == (3, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 1.0, 1.0]))
    assert (corpus.document_index.columns == expected_columns).all()

    corpus: VectorizedCorpus = trends_data.get_corpus(
        TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='decade')
    )
    assert corpus.data.shape == (2, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([8.0, 26.0]))
    assert (corpus.document_index.columns == expected_columns).all()

    corpus: VectorizedCorpus = trends_data.get_corpus(
        TrendsOpts(normalize=True, keyness=KeynessMetric.TF, group_by='decade')
    )
    assert corpus.data.shape == (2, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 1.0]))
    assert (corpus.document_index.columns == expected_columns).all()


# def test_trends_data_tf_idf():
#     trends_data: TrendsData = TrendsData().update(
#         corpus=simple_corpus(), corpus_folder='./tests/test_data', corpus_tag="dummy", n_count=100
#     )
#     corpus: VectorizedCorpus = trends_data.get_corpus(TrendsOpts(normalize=True, keyness=KeynessMetric.TF_IDF, group_by='year'))
#     assert corpus.data.shape == (9, 4)
#     # assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 1.0]))


def test_trends_data_top_terms():
    expected_category_column: str = 'time_period'
    trends_data: TrendsData = TrendsData().update(
        corpus=simple_corpus(), corpus_folder='./tests/test_data', corpus_tag="dummy", n_count=100
    )
    corpus = trends_data.get_corpus(TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='year'))
    assert expected_category_column in corpus.document_index

    n_count = 4
    df = corpus.get_top_terms(category_column=trends_data.category_column, n_count=n_count, kind='token')
    assert df is not None
    assert df.columns.tolist() == ['2009', '2013', '2014', '2017']
    assert df['2009'].tolist() == ['c', 'a', 'b', 'd']
    assert df['2013'].tolist() == ['c', 'a', 'b', '*']

    df = corpus.get_top_terms(category_column=trends_data.category_column, n_count=n_count, kind='token/count')
    assert df is not None
    assert df.columns.tolist() == ['2009', '2013', '2014', '2017']
    assert df['2009'].tolist() == ['c/4', 'a/2', 'b/1', 'd/1']
    assert df['2013'].tolist() == ['c/3', 'a/2', 'b/2', '*/0']

    df = corpus.get_top_terms(category_column=trends_data.category_column, n_count=n_count, kind='token+count')
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

    corpus = trends_data.get_corpus(TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='lustrum'))
    df = corpus.get_top_terms(category_column=trends_data.category_column, n_count=n_count, kind='token')
    assert df is not None
    assert df.columns.tolist() == ['2005', '2010', '2015']
    assert df['2005'].tolist() == ['c', 'a', 'b', 'd']
    assert df['2010'].tolist() == ['b', 'c', 'a', '*']

    corpus = trends_data.get_corpus(TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='decade'))
    df = corpus.get_top_terms(category_column=trends_data.category_column, n_count=n_count, kind='token')
    assert df is not None
    assert df.columns.tolist() == ['2000', '2010']
    assert df['2000'].tolist() == ['c', 'a', 'b', 'd']
    assert df['2010'].tolist() == ['b', 'a', 'c', 'd']

    corpus = trends_data.get_corpus(TrendsOpts(normalize=True, keyness=KeynessMetric.TF, group_by='decade'))
    df = corpus.get_top_terms(category_column=trends_data.category_column, n_count=n_count, kind='token+count')
    assert df is not None
    assert df.columns.tolist() == ['2000', '2000/Count', '2010', '2010/Count']
    assert np.allclose(
        df['2010/Count'].tolist(), [0.34615384615384615, 0.3076923076923077, 0.2692307692307693, 0.07692307692307693]
    )

    corpus = trends_data.get_corpus(TrendsOpts(normalize=True, keyness=KeynessMetric.TF_IDF, group_by='decade'))
    df = corpus.get_top_terms(category_column=trends_data.category_column, n_count=n_count, kind='token+count')
    assert df is not None
    assert df.columns.tolist() == ['2000', '2000/Count', '2010', '2010/Count']
    assert np.allclose(
        df['2010/Count'].tolist(), [0.3427448893375985, 0.29717678877203213, 0.2852926960827151, 0.07478562580765426]
    )


@patch('penelope.common.goodness_of_fit.compute_goddness_of_fits_to_uniform', lambda *_, **__: Mock(spec=pd.DataFrame))
@patch('penelope.common.goodness_of_fit.compile_most_deviating_words', lambda *_, **__: Mock(spec=pd.DataFrame))
@patch('penelope.common.goodness_of_fit.get_most_deviating_words', lambda *_, **__: Mock(spec=pd.DataFrame))
def test_group_by_year():
    corpus_folder = './tests/test_data/VENUS'
    corpus_tag = 'VENUS'

    corpus: VectorizedCorpus = Mock(spec=VectorizedCorpus)
    corpus = corpus.group_by_year()

    trends_data: TrendsData = TrendsData(
        corpus=corpus,
        corpus_folder=corpus_folder,
        corpus_tag=corpus_tag,
        n_count=100,
    ).update()

    assert trends_data is not None


def test_find_word_indices():
    trends_data = TrendsData().update(corpus=simple_corpus(), corpus_folder='.', corpus_tag='dummy')
    indices = trends_data.find_word_indices(
        TrendsOpts(group_by='year', normalize=False, smooth=False, keyness=KeynessMetric.TF, words=["c"], word_count=2)
    )
    assert indices == [2]


def test_find_words():
    pass
