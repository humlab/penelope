import unittest.mock as mock

import ipywidgets

from penelope.corpus import VectorizedCorpus
from penelope.notebook.word_trends import ITrendDisplayer, TrendsGUI, TrendsService


def mocked_displayer_ctor(**_):
    m = mock.MagicMock(ITrendDisplayer)
    m.name = "apa"
    m.titles = "apa"
    return m


def test_TrendsGUI_setup():
    gui = TrendsGUI().setup(displayers=[mocked_displayer_ctor])
    assert len(gui._displayers) == 1  # pylint: disable=protected-access


def test_TrendsGUI_layout():
    w = TrendsGUI().setup(displayers=[mocked_displayer_ctor]).layout()
    assert isinstance(w, ipywidgets.CoreWidget)


def test_TrendsGUI_display():
    corpus = mock.Mock(spec=VectorizedCorpus)
    trends_service = mock.MagicMock(spec=TrendsService, corpus=corpus, category_column="apa")
    gui = TrendsGUI().setup(displayers=[mocked_displayer_ctor])
    gui.trends_service = trends_service
    gui.display(trends_service=trends_service)
