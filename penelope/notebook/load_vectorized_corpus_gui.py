import glob
import os
import types
from typing import Callable

import ipywidgets
from penelope.corpus import VectorizedCorpus

# from penelope.utility.utils import right_chop, getLogger

# logger = getLogger('corpus_text_analysis')

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


def display_gui(data_folder: str, corpus_loaded_callback: Callable):

    corpus_suffix = '_vectorizer_data.pickle'
    corpus_files = sorted(glob.glob(os.path.join(data_folder, "*" + corpus_suffix)))
    corpus_tags = list(map(lambda x: right_chop(x, corpus_suffix), corpus_files))

    gui = types.SimpleNamespace(
        corpus_tag=ipywidgets.Dropdown(
            description='Corpus', options=corpus_tags, value=None, layout=ipywidgets.Layout(width='400px')
        ),
        button=ipywidgets.Button(
            description='Load',
            button_style='Success',
            layout=ipywidgets.Layout(width='115px', background_color='blue'),
        ),
        min_word_count=ipywidgets.IntSlider(description='Min Count', min=1, max=1000, step=1, value=1),
        output=ipywidgets.Output(),
    )

    def on_button_clicked(_):

        if gui.corpus_tag.value is None:
            return
        gui.button.disabled = True
        gui.output.clear_output()
        with gui.output:
            try:
                v_corpus = load_corpus(
                    data_folder, gui.corpus_tag.value, min_word_count=None, n_top=None, norm_axis=None
                )
                corpus_loaded_callback(v_corpus, gui.corpus_tag.value, gui.output)
            except Exception as ex:
                print(ex)
            finally:
                gui.button.disabled = False

    gui.button.on_click(on_button_clicked)

    return ipywidgets.VBox(
        [
            ipywidgets.HBox([ipywidgets.VBox([gui.corpus_tag, gui.min_word_count]), gui.button]),
            gui.output,
        ]
    )
