import numpy as np
import pytest
from penelope.common.curve_fit import pchip_spline, rolling_average_smoother
from penelope.notebook.word_trends import BarDisplayer, LineDisplayer, TableDisplayer
from penelope.notebook.word_trends.displayers._displayer import MultiLineDataMixin, PenelopeBugCheck, YearTokenDataMixin
from tests.utils import create_smaller_vectorized_corpus

BIGGER_CORPUS_FILENAME = './tests/test_data/riksdagens-protokoll.1950-1959.ak.sparv4.csv.zip'
OUTPUT_FOLDER = './tests/output'


def test_BarDisplayer_create():
    assert BarDisplayer() is not None


def test_LineDisplayer_create():
    assert LineDisplayer() is not None


def test_TableDisplayer_create():
    assert TableDisplayer() is not None


def xtest_loaded_callback():
    pass


def test_compile_multiline_data_with_no_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1]
    multiline_data = MultiLineDataMixin().compile(corpus, indices, smoothers=None)

    assert isinstance(multiline_data, dict)
    assert ["A", "B"] == multiline_data['label']
    assert all([(x == y).all() for x, y in zip([[2013, 2014], [2013, 2014]], multiline_data['xs'])])
    assert len(multiline_data['color']) == 2
    assert len(multiline_data['ys']) == 2
    assert all([np.allclose(x, y) for x, y in zip([[4.0, 6.0], [3.0, 7.0]], multiline_data['ys'])])


def test_compile_multiline_data_with_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1, 2, 3]
    smoothers = [pchip_spline, rolling_average_smoother('nearest', 3)]
    multiline_data = MultiLineDataMixin().compile(corpus, indices, smoothers=smoothers)

    assert isinstance(multiline_data, dict)
    assert ["A", "B", "C", "D"] == multiline_data['label']
    assert len(multiline_data['xs']) == 4
    assert len(multiline_data['ys']) == 4
    assert len(multiline_data['color']) == 4
    assert len(multiline_data['ys']) == 4
    assert len(multiline_data['xs'][0]) > 2  # interpolated coordinates added
    assert len(multiline_data['ys'][0]) == len(multiline_data['xs'][0])  # interpolated coordinates added


def test_compile_year_token_vector_data_when_corpus_is_grouped_by_year_succeeds():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1, 2, 3]
    data = YearTokenDataMixin().compile(corpus, indices)
    assert isinstance(data, dict)
    assert all(token in data.keys() for token in ["a", "b", "c", "d"])
    assert len(data["b"]) == 2


def test_compile_year_token_vector_data_when_corpus_is_not_grouped_by_year_fails():
    corpus = create_smaller_vectorized_corpus()
    indices = [0, 1, 2, 3]
    with pytest.raises(PenelopeBugCheck):
        _ = YearTokenDataMixin().compile(corpus, indices)
