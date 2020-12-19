from unittest.mock import MagicMock, Mock, patch

import pandas as pd
from penelope.corpus import VectorizedCorpus
from penelope.notebook.co_occurrence.explore_co_occurrence_gui import ExploreCoOccurrencesGUI
from penelope.notebook.word_trends import TrendsData


def trends_data_mock() -> TrendsData:
    return Mock(
        spec=TrendsData,
        **{
            'corpus': MagicMock(spec=VectorizedCorpus),
            'goodness_of_fit': MagicMock(spec=pd.DataFrame),
            'most_deviating_overview': MagicMock(spec=pd.DataFrame),
            'most_deviating': MagicMock(spec=pd.DataFrame),
        },
    )


# @patch('penelope.notebook.utility.OutputsTabExt')
# @patch('penelope.notebook.co_occurrence.explore_co_occurrence_gui.ipywidgets', Mock())
# def test_ExploreCoOccurrencesGUI_create_and_layout(tab):

#     trends_data: TrendsData = trends_data_mock()

#     gui: ExploreCoOccurrencesGUI = ExploreCoOccurrencesGUI(trends_data=trends_data)

#     layout = gui.layout()

#     # assert layout is not None
