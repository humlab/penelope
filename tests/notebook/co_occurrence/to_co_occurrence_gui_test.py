from unittest.mock import Mock, patch

import penelope.notebook.co_occurrence.to_co_occurrence_gui as to_co_occurrence_gui
import penelope.notebook.utility as notebook_utility
from penelope.pipeline.config import CorpusConfig, CorpusType
from penelope.utility.pos_tags import PoS_Tag_Scheme


def dummy_config():
    return Mock(
        spec=CorpusConfig,
        **{
            'pipeline_payload.source': 'dummy',
            'corpus_type': CorpusType.Pipeline,
            'text_reader_opts.filename_fields': [],
        },
    )


@patch(
    'penelope.utility.get_pos_schema', return_value=Mock(spec=PoS_Tag_Scheme, **{'groups': {'Noun': [], 'Verb': []}})
)
@patch('penelope.notebook.utility.FileChooserExt2', Mock(spec=notebook_utility.FileChooserExt2))
def test_to_co_occurrence_create_gui(z):  # pylint: disable=unused-argument
    def done_callback(_: to_co_occurrence_gui.GUI):
        pass

    def compute_callback(corpus_config, args, partition_key, done_callback):  # pylint: disable=unused-argument
        pass

    corpus_config = dummy_config()

    gui = to_co_occurrence_gui.create_gui(
        corpus_folder='./tests/test_data',
        corpus_config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    assert gui is not None


@patch(
    'penelope.utility.get_pos_schema', return_value=Mock(spec=PoS_Tag_Scheme, **{'groups': {'Noun': [], 'Verb': []}})
)
@patch('penelope.notebook.utility.FileChooserExt2', Mock(spec=notebook_utility.FileChooserExt2))
def test_GUI_setup(z):  # pylint: disable=unused-argument
    def compute_callback(corpus_config, args, partition_key, done_callback):  # pylint: disable=unused-argument
        pass

    corpus_config = dummy_config()
    gui = to_co_occurrence_gui.GUI(
        default_corpus_path='./tests/test_data',
        default_corpus_filename='',
        default_target_folder='./tests/output',
    ).setup(
        config=corpus_config,
        compute_callback=compute_callback,
    )

    # layout = gui.layout()
    # gui._compute_handler({})  # pylint: disable=protected-access

    assert gui is not None
