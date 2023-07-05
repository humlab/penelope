# from unittest import mock

# from penelope.corpus import VectorizedCorpus, load_corpus
from penelope.notebook.dtm import LoadGUI, create_load_gui

# pylint: disable=protected-access


def dummy_callback(*_, **__):
    pass


def test_gui_setup():
    gui = LoadGUI(folder='./tests/test_data', filename_pattern='*.*', load_callback=None, done_callback=None)
    assert gui.setup() is not None


def test_gui_layout():
    gui = LoadGUI(folder='./tests/test_data', filename_pattern='*.*', load_callback=None, done_callback=None)
    assert gui.setup().layout() is not None


def test_create_gui():
    gui = create_load_gui(
        corpus_folder='./tests/test_data',
        loaded_callback=dummy_callback,
    )
    assert gui is not None


# def test_gui_load_bug():

#     source_filename = (
#         "/data/inidun/courier/word_trends/Courier_allpos_nolemma_tf1/Courier_allpos_nolemma_tf1_vector_data.npz"
#     )

#     def load_corpus_callback(folder: str, tag: str) -> VectorizedCorpus:

#         corpus: VectorizedCorpus = load_corpus(
#             folder=folder, tag=tag, tf_threshold=None, n_top=None, axis=None, group_by_year=False
#         )

#         return corpus

#     loaded_corpus_callback = mock.MagicMock()

#     with mock.patch('penelope.notebook.dtm.LoadGUI.corpus_filename', new_callable=mock.PropertyMock) as mocked_method:
#         mocked_method.return_value = source_filename

#         gui = LoadGUI(
#             folder='/data/inidun/courier/word_trends',
#             filename_pattern='*_vector_data.npz',
#             load_callback=load_corpus_callback,
#             done_callback=loaded_corpus_callback,
#         )

#         gui.setup()
#         gui.layout()

#         assert gui.corpus_filename == source_filename

#         gui._load_handler({})

#         assert loaded_corpus_callback.called
