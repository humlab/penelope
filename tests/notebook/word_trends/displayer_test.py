import io

import ipycytoscape
import numpy as np
import pandas as pd
import pytest
from penelope import co_occurrence
from penelope.common.curve_fit import pchip_spline, rolling_average_smoother
from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends import BundleTrendsData
from penelope.notebook.word_trends.displayers import (
    BarDisplayer,
    UnstackedDataMixin,
    ITrendDisplayer,
    LineDisplayer,
    StackedDataMixin,
    NetworkDisplayer,
    PenelopeBugCheck,
    TableDisplayer,
    TopTokensDisplayer,
    UnnestedExplodeTableDisplayer,
    UnnestedTableDisplayer,
    create_network,
)
from penelope.notebook.word_trends.displayers.display_top_table import CoOccurrenceTopTokensDisplayer
from tests.fixtures import create_smaller_vectorized_corpus

BIGGER_CORPUS_FILENAME = './tests/test_data/riksdagens-protokoll.1950-1959.ak.sparv4.csv.zip'
OUTPUT_FOLDER = './tests/output'

# pylint: disable=redefined-outer-name


def test_BarDisplayer_create():
    assert BarDisplayer() is not None


def test_LineDisplayer_create():
    assert LineDisplayer() is not None


def test_TableDisplayer_create():
    assert TableDisplayer() is not None


def xtest_loaded_callback():
    pass


def test_compile_multiline_data_with_no_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year(target_column_name="category")
    indices = [0, 1]
    multiline_data = StackedDataMixin().compile(corpus, indices, smoothers=None)

    assert isinstance(multiline_data, dict)
    assert ["A", "B"] == multiline_data['label']
    assert all((x == y).all() for x, y in zip([[2013, 2014], [2013, 2014]], multiline_data['xs']))
    assert len(multiline_data['color']) == 2
    assert len(multiline_data['ys']) == 2
    assert all(np.allclose(x, y) for x, y in zip([[4.0, 6.0], [3.0, 7.0]], multiline_data['ys']))


def test_compile_multiline_data_with_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year(target_column_name="category")
    indices = [0, 1, 2, 3]
    smoothers = [pchip_spline, rolling_average_smoother('nearest', 3)]
    multiline_data = StackedDataMixin().compile(corpus, indices, smoothers=smoothers)

    assert isinstance(multiline_data, dict)
    assert ["A", "B", "C", "D"] == multiline_data['label']
    assert len(multiline_data['xs']) == 4
    assert len(multiline_data['ys']) == 4
    assert len(multiline_data['color']) == 4
    assert len(multiline_data['ys']) == 4
    assert len(multiline_data['xs'][0]) > 2  # interpolated coordinates added
    assert len(multiline_data['ys'][0]) == len(multiline_data['xs'][0])  # interpolated coordinates added


def test_compile_year_token_vector_data_when_corpus_is_grouped_by_year_succeeds():
    corpus = create_smaller_vectorized_corpus().group_by_year(target_column_name="category")
    indices = [0, 1, 2, 3]
    data = UnstackedDataMixin().compile(corpus, indices)
    assert isinstance(data, dict)
    assert all(token in data.keys() for token in ["a", "b", "c", "d"])
    assert len(data["b"]) == 2


def test_compile_year_token_vector_data_when_corpus_is_not_grouped_by_year_fails():
    corpus = create_smaller_vectorized_corpus()
    indices = [0, 1, 2, 3]
    with pytest.raises(PenelopeBugCheck):
        _ = UnstackedDataMixin().compile(corpus, indices)


@pytest.fixture(scope="module")
def bundle() -> co_occurrence.Bundle:
    folder, tag = './tests/test_data/SSI', 'SSI'
    filename = co_occurrence.to_filename(folder=folder, tag=tag)
    bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename, compute_frame=False)
    return bundle


@pytest.fixture(scope="module")
def trends_data(bundle) -> BundleTrendsData:
    trends_data: BundleTrendsData = BundleTrendsData(bundle=bundle)
    return trends_data


@pytest.mark.parametrize(
    'displayer_cls',
    [
        BarDisplayer,
        LineDisplayer,
        TableDisplayer,
        UnnestedExplodeTableDisplayer,
        UnnestedTableDisplayer,
        NetworkDisplayer,
    ],
)
def test_displayer_compile_and_display(displayer_cls, bundle: co_occurrence.Bundle):

    corpus: VectorizedCorpus = bundle.corpus.group_by_year(target_column_name='category')

    displayer: ITrendDisplayer = displayer_cls()
    displayer.setup()

    plot_data: dict = displayer.compile(corpus=corpus, indices=[0, 1, 2])

    assert plot_data is not None


def test_top_tokens_displayer_compile_and_display(bundle: co_occurrence.Bundle):

    displayer: ITrendDisplayer = TopTokensDisplayer(corpus=bundle.corpus)
    displayer.setup()

    plot_data: dict = displayer.compile()

    assert plot_data is not None


def test_co_occurrence_top_tokens_displayer_compile_and_display(bundle: co_occurrence.Bundle):

    displayer: ITrendDisplayer = CoOccurrenceTopTokensDisplayer(bundle=bundle)
    displayer.setup()

    plot_data: dict = displayer.compile()

    assert plot_data is not None


def test_network():

    data_str = """;year;token;count;w1;w2
0;1920;riksdag/ledamot;18;riksdag;ledamot
1;1930;riksdag/ledamot;22;riksdag;ledamot
2;1940;riksdag/ledamot;29;riksdag;ledamot
3;1950;riksdag/ledamot;29;riksdag;ledamot
4;1960;riksdag/ledamot;30;riksdag;ledamot
5;1970;riksdag/ledamot;22;riksdag;ledamot
6;1980;riksdag/ledamot;582;riksdag;ledamot
7;1990;riksdag/ledamot;640;riksdag;ledamot
8;2000;riksdag/ledamot;710;riksdag;ledamot"""

    df = pd.read_csv(io.StringIO(data_str), sep=';', index_col=0)

    w: ipycytoscape.CytoscapeWidget = create_network(df, category_column_name='year')

    assert w is not None
