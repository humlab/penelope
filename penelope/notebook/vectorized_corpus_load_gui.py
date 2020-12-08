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

    default_corpus_folder: str
    filename_pattern: str
    load_callback: Callable

    corpus_filename: ipyfilechooser.FileChooser = None

    button = ipywidgets.Button(
        description='Load',
        button_style='Success',
        layout=ipywidgets.Layout(width='115px', background_color='blue'),
    )

    def _load_handler(self, _):

        # self.output.clear_output()

        # with self.output:
        #     try:

        if (self.corpus_filename.selected or "") == "":
            print("Please select a corpus")
            return

            # raise ValueError("Please select a corpus")
        self.button.disabled = True

        input_filename = self.corpus_filename.selected

        if not os.path.isfile(input_filename):
            print("Please sSelect a file")
            return

        input_folder, filename = os.path.split(input_filename)
        corpus_tag = right_chop(filename, self.filename_pattern[1:])

        self.load_callback(
            corpus_folder=input_folder,
            corpus_tag=corpus_tag,
        )

        # except (ValueError, FileNotFoundError, Exception) as ex:
        #     logger.error(ex)
        #     raise
        # finally:
        self.button.disabled = False

    def setup(self):

        self.corpus_filename: ipyfilechooser.FileChooser = ipyfilechooser.FileChooser(
            path=self.default_corpus_folder or default_data_folder(),
            filter_pattern=self.filename_pattern,
            title='<b>Corpus file (vectorized corpus)</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False,
        )
        self.button.on_click(self._load_handler)
        return self
        # shorten_filechooser_label(self.corpus_filename, 50)

    def layout(self):
        return ipywidgets.VBox(
            [
                ipywidgets.HBox([ipywidgets.VBox([self.corpus_filename]), self.button]),
            ]
        )


def display_gui(
    *,
    corpus_folder: str,
    loaded_callback: Callable,
):
    print("INSIDE vectorized_corpus_load_gui.display_gui")

    filename_pattern = '*_vectorizer_data.pickle'

    def load_corpus_callback(corpus_folder: str, corpus_tag: str):
        corpus = load_corpus(corpus_folder, corpus_tag, min_word_count=None, n_top=None, norm_axis=None)
        loaded_callback(
            corpus_folder=corpus_folder,
            corpus_tag=corpus_tag,
            corpus=corpus,
        )

    gui = GUI(
        default_corpus_folder=corpus_folder,
        filename_pattern=filename_pattern,
        load_callback=load_corpus_callback,
    ).setup()

    return gui
