from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.notebook.word_trends import TrendsData
from tests.utils import OUTPUT_FOLDER


def simple_corpus():
    corpus = VectorizedCorpus(
        np.array(
            [
                [2, 1, 4, 1],
                [2, 2, 3, 0],
                [2, 3, 2, 0],
                [2, 4, 1, 1],
                [2, 0, 1, 1],
            ]
        ),
        {'a': 0, 'b': 1, 'c': 2, 'd': 3},
        pd.DataFrame(
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

    trends_data: TrendsData = TrendsData().update(
        corpus=simple_corpus(),
        corpus_folder='./tests/test_data',
        corpus_tag="dummy",
        n_count=100,
    )

    corpus: VectorizedCorpus = trends_data.get_corpus(False, 'year')
    assert corpus.data.shape == (9, 4)  # Shape of 'year' should include years without documents (gaps are filled)
    assert corpus.data.shape == trends_data.corpus.data.shape
    assert corpus.data.sum() == trends_data.corpus.data.sum()
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([8.0, 0.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0, 12.0]))
    assert 'year' in corpus.document_index.columns
    assert 'category' in corpus.document_index.columns

    corpus: VectorizedCorpus = trends_data.get_corpus(True, 'year')
    assert corpus.data.shape == (9, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]))
    assert 'year' in corpus.document_index.columns
    assert 'category' in corpus.document_index.columns

    expected_columns = ['category', 'filename', 'document_name', 'year_min', 'year_max', 'year_size']

    corpus: VectorizedCorpus = trends_data.get_corpus(False, 'lustrum')
    assert corpus.data.shape == (3, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([8.0, 14.0, 12.0]))
    assert (corpus.document_index.columns == expected_columns).all()

    corpus: VectorizedCorpus = trends_data.get_corpus(True, 'lustrum')
    assert corpus.data.shape == (3, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 1.0, 1.0]))
    assert (corpus.document_index.columns == expected_columns).all()

    corpus: VectorizedCorpus = trends_data.get_corpus(False, 'decade')
    assert corpus.data.shape == (2, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([8.0, 26.0]))
    assert (corpus.document_index.columns == expected_columns).all()

    corpus: VectorizedCorpus = trends_data.get_corpus(True, 'decade')
    assert corpus.data.shape == (2, 4)
    assert np.allclose(corpus.data.sum(axis=1).A1, np.array([1.0, 1.0]))
    assert (corpus.document_index.columns == expected_columns).all()


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
