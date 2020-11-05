import glob
import os
import types
from typing import Callable, List, Set

import ipywidgets as widgets
from penelope.utility import flatten, strip_paths
from penelope.utility.tags import SUC_PoS_tag_groups
from penelope.workflows import execute_workflow_concept_co_occurrence

# from penelope.utility.utils import right_chop, getLogger

# logger = getLogger('corpus_text_analysis')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


def generate_concept_co_occurrences(
    input_filename: str,
    output_filename: str,
    *,
    concept: Set[str],
    context_width: int,
    pos_includes: List[str],
    count_threshold: int = 1,
    no_concept: bool = False,
    lemmatize: bool = True,
    to_lowercase: bool = True,
    remove_stopwords: str = None,
):
    """[summary]

    Parameters
    ----------
    input_filename : str
        [description]
    output_filename : str
        [description]
    concept : Set[str]
        [description]
    context_width : int
        [description]
    pos_includes : List[str]
        [description]
    count_threshold : int, optional
        [description], by default 1
    no_concept : bool, optional
        [description], by default False
    lemmatize : bool, optional
    to_lowercase : bool, optional
    remove_stopwords : bool, optional
    """
    execute_workflow_concept_co_occurrence(
        input_filename=input_filename,
        output_filename=output_filename,
        concept=concept,
        no_concept=no_concept,
        count_threshold=count_threshold if count_threshold is not None and count_threshold > 1 else None,
        context_width=context_width,
        pos_includes=f"|{'|'.join(pos_includes)}|",
        lemmatize=lemmatize,
        to_lowercase=to_lowercase,
        remove_stopwords='swedish' if remove_stopwords else None,
        # keep_symbols = True,
        # keep_numerals = True,
        # only_alphabetic = False,
        # only_any_alphanumeric = False,
        pos_excludes="|MAD|MID|PAD|",
        min_word_length=2,
        partition_keys="year",
        filename_field={"year": r"prot\_(\d{4}).*"},
        store_vectorized=True,
    )


def display_gui(data_folder: str, corpus_pattern: str, generated_callback: Callable):
    lw = lambda w: widgets.Layout(width=w)
    corpus_filenames = list(map(strip_paths, sorted(glob.glob(os.path.join(data_folder, corpus_pattern)))))

    gui = types.SimpleNamespace(
        input_filename=widgets.Dropdown(description='Corpus', options=corpus_filenames, value=None, layout=lw('400px')),
        output_filename=widgets.Text(
            value='',
            placeholder='Enter a filename without extension',
            description='Result',
            disabled=False,
            layout=lw('400px'),
        ),
        concept=widgets.Text(
            value='',
            placeholder='Use comma (,) as word delimiter',
            description='Concept',
            disabled=False,
            layout=lw('400px'),
        ),
        pos_includes=widgets.SelectMultiple(
            options=SUC_PoS_tag_groups,
            value=[SUC_PoS_tag_groups['Noun'], SUC_PoS_tag_groups['Verb']],
            rows=8,
            description='PoS',
            disabled=False,
            layout=lw('400px'),
        ),
        context_width=widgets.IntSlider(description='Context Size', min=1, max=20, step=1, value=2, layout=lw('400px')),
        count_threshold=widgets.IntSlider(
            description='Min Count', min=1, max=1000, step=1, value=1, layout=lw('400px')
        ),
        no_concept=widgets.ToggleButton(value=True, description='No Concept', icon='check', layout=lw('140px')),
        lemmatize=widgets.ToggleButton(value=True, description='Lemmatize', icon='check', layout=lw('140px')),
        to_lowercase=widgets.ToggleButton(value=True, description='To Lower', icon='check', layout=lw('140px')),
        remove_stopwords=widgets.ToggleButton(value=True, description='No Stopwords', icon='check', layout=lw('140px')),
        button=widgets.Button(
            description='Load',
            button_style='Success',
            layout=lw('140px'),
        ),
        output=widgets.Output(),
    )

    def on_button_clicked(_):

        if gui.input_filename.value is None:
            return

        if gui.output_filename.value.strip() == "":
            return

        with gui.output:

            # gui.output.clear_output()
            gui.button.disabled = True

            concept = set(map(str.strip, gui.concept.value.split(',')))

            generate_concept_co_occurrences(
                input_filename=gui.input_filename.value,
                output_filename=gui.output_filename.value,
                concept=concept,
                context_width=gui.context_width.value,
                pos_includes=flatten(gui.pos_includes.value),
                count_threshold=gui.count_threshold.value,
                no_concept=gui.no_concept.value,
                lemmatize=gui.lemmatize.value,
                to_lowercase=gui.to_lowercase.value,
                remove_stopwords=gui.remove_stopwords.value,
            )

            if generated_callback is not None:
                generated_callback(gui.output_filename.value, gui.output)

            gui.button.disabled = False

    gui.button.on_click(on_button_clicked)

    return widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            gui.input_filename,
                            gui.output_filename,
                            gui.concept,
                            gui.pos_includes,
                            gui.context_width,
                            gui.count_threshold,
                        ]
                    ),
                    widgets.VBox([gui.no_concept, gui.lemmatize, gui.to_lowercase, gui.remove_stopwords, gui.button]),
                ]
            ),
            gui.output,
        ]
    )
