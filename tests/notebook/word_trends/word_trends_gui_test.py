from unittest.mock import Mock

import ipywidgets
from penelope.notebook.word_trends import TrendsGUI
from penelope.notebook.word_trends.displayers import ITrendDisplayer
from penelope.notebook.word_trends.word_trend_data import WordTrendData


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
    trend_data = Mock(spec=WordTrendData)
    displayer = Mock(ITrendDisplayer)
    gui = TrendsGUI().setup(displayers=[displayer])
    gui.display(trend_data=trend_data)
    assert gui.current_displayer.compile.call_count == 1
    assert gui.current_displayer.plot.call_count == 1
