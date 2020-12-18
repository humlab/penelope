
import pandas as pd
from tests.utils import OUTPUT_FOLDER, create_smaller_vectorized_corpus
from penelope.notebook.word_trends import WordTrendData

def test_WordTrendData_create():
    pass

def test_WordTrendData_update():

    corpus = create_smaller_vectorized_corpus()
    n_count = 10000

    data = WordTrendData().update(
        corpus=corpus,
        corpus_folder=OUTPUT_FOLDER,
        corpus_tag="dummy",
        n_count=n_count,
    )

    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating_overview, pd.DataFrame)
    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating, pd.DataFrame)

def test_WordTrendData_remember():
    pass

def test_WordTrendData_get_corpus():
    pass
