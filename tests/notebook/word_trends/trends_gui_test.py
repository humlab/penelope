from unittest.mock import MagicMock, Mock

import ipywidgets
from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends import ITrendDisplayer, TrendsData, TrendsGUI


def test_TrendsGUI_setup():
    displayer = Mock(ITrendDisplayer)
    gui = TrendsGUI().setup(displayers=[displayer])
    assert displayer.call_count == 1
    assert len(gui._displayers) == 1  # pylint: disable=protected-access


def test_TrendsGUI_layout():
    displayer = Mock(ITrendDisplayer)
    w = TrendsGUI().setup(displayers=[displayer]).layout()
    assert isinstance(w, ipywidgets.CoreWidget)


def test_TrendsGUI_display():
    corpus = Mock(spec=VectorizedCorpus)
    trends_data = MagicMock(spec=TrendsData, corpus=corpus, category_column="apa")
    displayer = Mock(ITrendDisplayer)
    gui = TrendsGUI().setup(displayers=[displayer])
    gui.display(trends_data=trends_data)
    assert gui.current_displayer.display.call_count == 1
