import os
from dataclasses import dataclass
from typing import Any, Callable

import ipyfilechooser
import ipywidgets
from penelope.corpus import VectorizedCorpus
from penelope.utility import default_data_folder, getLogger

logger = getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes
def right_chop(s: str, suffix: str) -> str:
    """Returns `s` with `suffix` removed"""
    return s[: -len(suffix)] if suffix != "" and s.endswith(suffix) else s


def load_corpus(
    folder: str, tag: str, min_word_count: int = None, n_top: int = None, norm_axis: int = None
) -> VectorizedCorpus:
    """Loads and returns a vectorized corpus from `folder` with tag `tag`

    Parameters
    ----------
    folder : str
        Folder in which the corpus files exist
    tag : str
        Corpus tag i.e. the prefix preceeding the suffix '_vectorizer_data.pickle'
    min_word_count : int, optional
        If specified then tokens below given threshold count are stripped away, by default None
    n_top : int, optional
        [description], by default 0
    norm_axis : int, optional
        Specifies normalization, 0: over year, 1: token, 2: both by default None

    Returns
    -------
    VectorizedCorpus
    """
    # n_count, n_top, axis, keep_magnitude = None, None, 1, False
    v_corpus = VectorizedCorpus.load(tag=tag, folder=folder).group_by_year()

    if min_word_count is not None and min_word_count > 1:
        v_corpus = v_corpus.slice_by_n_count(min_word_count)

    if n_top is not None:
        v_corpus = v_corpus.slice_by_n_top(n_top)

    if norm_axis in (1, 2) and v_corpus.data.shape[1] > 0:
        v_corpus = v_corpus.normalize(axis=1)

    if norm_axis in (0, 2):
        v_corpus = v_corpus.normalize(axis=0)

    return v_corpus


@dataclass
class GUI:
    input_filename_chooser = ipyfilechooser.FileChooser(
        path=default_data_folder(),
        filter_pattern='*_vectorizer_data.pickle',
        title='<b>Corpus file (vectorized corpus)</b>',
        show_hidden=False,
        select_default=False,
        use_dir_icons=True,
        show_only_dirs=False,
    )
    button = ipywidgets.Button(
        description='Load',
        button_style='Success',
        layout=ipywidgets.Layout(width='115px', background_color='blue'),
    )
    output = ipywidgets.Output()


def display_gui(*, loaded_callback: Callable[[ipywidgets.Output, str, str, VectorizedCorpus, Any], None] = None):

    corpus_suffix = '_vectorizer_data.pickle'

    gui = GUI()

    def on_button_clicked(_):

        gui.output.clear_output()
        gui.button.disabled = True

        with gui.output:
            try:

                if (gui.input_filename_chooser.selected or "") == "":
                    raise ValueError("Please select a corpus")

                input_filename = gui.input_filename_chooser.selected
                input_folder, filename = os.path.split(input_filename)
                corpus_tag = right_chop(filename, corpus_suffix)
                v_corpus = load_corpus(input_folder, corpus_tag, min_word_count=None, n_top=None, norm_axis=None)

                if loaded_callback is not None:
                    loaded_callback(
                        corpus_folder=input_folder, corpus_tag=corpus_tag, corpus=v_corpus, output=gui.output
                    )

            except (ValueError, FileNotFoundError, Exception) as ex:
                logger.error(ex)
            finally:
                gui.button.disabled = False

    gui.button.on_click(on_button_clicked)

    return ipywidgets.VBox(
        [
            ipywidgets.HBox([ipywidgets.VBox([gui.input_filename_chooser]), gui.button]),
            gui.output,
        ]
    )
