import unittest
from unittest import mock
from unittest.mock import Mock

import penelope.notebook.co_occurrence.load_co_occurrences_gui as load_co_occurrences_gui
import penelope.notebook.co_occurrence.to_co_occurrence_gui as to_co_occurrence_gui
import penelope.notebook.utility
from penelope.notebook.word_trends.gof_and_trends_gui import GofTrendsGUI
from penelope.pipeline.config import CorpusConfig, CorpusType
from penelope.utility.pos_tags import PoS_Tag_Scheme

DATA_FOLDER = './tests/test_data'


def test_load_co_occurrences_gui_create_gui():
    def load_callback(_: load_co_occurrences_gui.GUI):
        pass

    gui = load_co_occurrences_gui.create_gui(data_folder=DATA_FOLDER)

    gui = gui.setup(filename_pattern="*.zip", load_callback=load_callback)

    assert gui is not None

    layout = gui.layout()

    assert layout is not None


@unittest.mock.patch(
    'penelope.utility.get_pos_schema', return_value=Mock(spec=PoS_Tag_Scheme, **{'groups': {'Noun': [], 'Verb': []}})
)
@unittest.mock.patch('penelope.notebook.utility.FileChooserExt2', Mock(spec=penelope.notebook.utility.FileChooserExt2))
def test_to_co_occurrences_gui_create_gui(z):  # pylint: disable=unused-argument
    def done_callback(_: to_co_occurrence_gui.GUI):
        pass

    def compute_callback(corpus_config, args, partition_key, done_callback):  # pylint: disable=unused-argument
        pass

    corpus_config = Mock(
        spec=CorpusConfig,
        **{
            'pipeline_payload.source': 'dummy',
            'corpus_type': CorpusType.Pipeline,
            'text_reader_opts.filename_fields': [],
        },
    )

    gui = to_co_occurrence_gui.create_gui(
        corpus_folder=DATA_FOLDER,
        corpus_config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    assert gui is not None

    # layout = gui.layout()
    # assert layout is not None


@mock.patch(
    'penelope.notebook.utility.OutputsTabExt',
    mock.MagicMock(spec=penelope.notebook.utility.OutputsTabExt),
)
def test_trends_gui_create_gui():

    # word_trend_data: WordTrendData = mock.MagicMock(
    #     spec=WordTrendData, **{'most_deviating_overview.__getitem__': mock.Mock()}
    # )

    gui = GofTrendsGUI()

    assert gui is not None
    assert gui.layout() is not None
