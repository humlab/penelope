from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from penelope.corpus.dtm import VectorizedCorpus
from penelope.notebook.utility import OutputsTabExt
from penelope.notebook.word_trends import GofTrendsGUI, TrendsData, TrendsGUI
from penelope.notebook.word_trends.gof_and_trends_gui import GoFsGUI


def trends_data_mock():
    return Mock(
        spec=TrendsData,
        **{
            'corpus': MagicMock(spec=VectorizedCorpus),
            'goodness_of_fit': MagicMock(spec=pd.DataFrame),
            'most_deviating_overview': MagicMock(spec=pd.DataFrame),
            'most_deviating': MagicMock(spec=pd.DataFrame),
        },
    )


def test_GofsGUI_create():
    tab = Mock(OutputsTabExt)
    gui = GoFsGUI(tab_gof=tab)
    assert gui.tab_gof is tab


@patch('penelope.notebook.utility.OutputsTabExt')
def test_GoFsGUI_setup(tab):
    gui = GoFsGUI().setup()
    assert gui is not None
    assert tab.call_count == 1


@patch('penelope.notebook.utility.OutputsTabExt')
def test_GoFsGUI_layout(_):
    layout = GoFsGUI().setup().layout()
    assert layout is not None


@patch('penelope.notebook.utility.OutputsTabExt')
def test_GoFsGUI_display(tab):
    tab = Mock(OutputsTabExt)
    gui = GoFsGUI(tab_gof=tab)
    most_deviating_overview = MagicMock(spec=pd.DataFrame)
    trends_data = Mock(spec=TrendsData, most_deviating_overview=most_deviating_overview)
    gui.display(trends_data=trends_data)
    assert gui.is_displayed


@patch('penelope.notebook.utility.OutputsTabExt')
def test_GofTrendsGUI_layout(tab):

    gui = GofTrendsGUI(
        gofs_gui=Mock(spec=GoFsGUI),
        trends_gui=Mock(spec=TrendsGUI),
    )

    _ = gui.layout()

    assert tab.call_count == 1


def test_GofTrendsGUI_display():
    trends_data = Mock(spec=TrendsData)
    gofs_gui = Mock(spec=GoFsGUI)
    trends_gui = Mock(spec=TrendsGUI)
    gui = GofTrendsGUI(
        gofs_gui=gofs_gui,
        trends_gui=trends_gui,
    )
    gui.display(trends_data)

    assert gui.gofs_gui.display.call_count == 1
    assert gui.trends_gui.display.call_count == 1


def test_update_plot():  # gui: TrendsGUI, trends_data: TrendsData):
    pass


def test_create_gof_and_trends_gui():  # trends_data: TrendsData = None):
    pass
