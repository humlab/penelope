import ipywidgets
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.notebook.word_trends.word_trend_data import WordTrendData
from IPython.core.display import display
from penelope.notebook.word_trends import TrendsGUI
from unittest.mock import Mock
from penelope.notebook.word_trends.displayers import  ITrendDisplayer

def test_TrendsGUI_setup():
    displayer = Mock(ITrendDisplayer)
    gui = TrendsGUI().setup(update_handler=None, displayers=[displayer])
    assert displayer.call_count == 1
    assert len(gui._displayers) == 1  # pylint: disable=protected-access

def test_TrendsGUI_layout():
    displayer = Mock(ITrendDisplayer)
    w = TrendsGUI().setup(update_handler=None, displayers=[displayer]).layout()
    assert isinstance(w, ipywidgets.CoreWidget)

def test_TrendsGUI_display():
    trend_data = Mock(spec=WordTrendData)
    corpus = Mock(spec=VectorizedCorpus)
    displayer = Mock(ITrendDisplayer)
    indices = [0]
    gui = TrendsGUI().setup(update_handler=None, displayers=[displayer])
    gui.display(trend_data=trend_data, corpus=corpus, indices=indices)
    assert gui.current_displayer.compile.call_count == 1
    assert gui.current_displayer.plot.call_count == 1


