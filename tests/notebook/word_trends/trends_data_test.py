import pandas as pd
from penelope.notebook.word_trends import TrendsData
from tests.utils import OUTPUT_FOLDER, create_smaller_vectorized_corpus


def test_TrendsData_create():
    pass


def test_TrendsData_update():

    corpus = create_smaller_vectorized_corpus()
    n_count = 10000

    data = TrendsData().update(
        corpus=corpus,
        corpus_folder=OUTPUT_FOLDER,
        corpus_tag="dummy",
        n_count=n_count,
    )

    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating_overview, pd.DataFrame)
    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating, pd.DataFrame)


def test_TrendsData_remember():
    pass


def test_TrendsData_get_corpus():
    pass
