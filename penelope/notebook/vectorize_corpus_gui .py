import glob
import os
import types
from typing import Callable

import ipywidgets as widgets
from penelope.corpus.readers.annotation_opts import AnnotationOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.utility import flatten, strip_paths
from penelope.utility.tags import SUC_PoS_tag_groups
from penelope.workflows import vectorize_sparv_csv_corpus_workflow, vectorize_tokenized_corpus_workflow

# from penelope.utility.utils import right_chop, getLogger

# logger = getLogger('corpus_text_analysis')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


def display_gui(data_folder: str, corpus_pattern: str, generated_callback: Callable):
    lw = lambda w: widgets.Layout(width=w)
    corpus_filenames = list(map(strip_paths, sorted(glob.glob(os.path.join(data_folder, corpus_pattern)))))

    gui = types.SimpleNamespace(
        input_filename=widgets.Dropdown(description='Corpus', options=corpus_filenames, value=None, layout=lw('400px')),
        corpus_type=widgets.Dropdown(
            description='Corpus', options=['text', 'sparv4-csv'], value='sparv4-csv', layout=lw('400px')
        ),
        output_tag=widgets.Text(
            value='',
            placeholder='Enter a unique output filename prefix',
            description='Output tag',
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

        output_folder = data_folder
        if gui.output_tag.strip() == "":
            print("please specify a unique string that will be prepended to output file")
            return

        input_filename = os.path.join(data_folder, gui.input_filename.value)

        if not os.path.isfile(input_filename):
            raise FileNotFoundError(input_filename)

        with gui.output:

            # gui.output.clear_output()
            gui.button.disabled = True

            tokens_transform_opts = TokensTransformOpts(
                remove_stopwords=gui.remove_stopwords.value,
                to_lowercase=gui.to_lowercase.value,
            )

            if gui.corpus_type.value == 'sparv4-csv':
                annotation_opts = AnnotationOpts(
                    pos_includes=f"|{'|'.join(flatten(gui.pos_includes.value))}|",
                    pos_excludes="|MAD|MID|PAD|",
                    lemmatize=gui.lemmatize.value,
                )
                vectorize_sparv_csv_corpus_workflow(
                    input_filename=input_filename,
                    output_folder=output_folder,
                    output_tag=gui.output_tag.strip(),
                    count_threshold=gui.count_threshold.value,
                    annotation_opts=annotation_opts,
                    tokens_transform_opts=tokens_transform_opts,
                )

            elif gui.corpus_type.value == 'text':

                vectorize_tokenized_corpus_workflow(
                    input_filename=input_filename,
                    output_folder=output_folder,
                    output_tag=gui.output_tag.strip(),
                    count_threshold=gui.count_threshold.value,
                    tokens_transform_opts=tokens_transform_opts,
                )

            if generated_callback is not None:
                generated_callback(gui.output_tag.value, gui.output)

            gui.button.disabled = False

    def corpus_type_changed(*_):
        gui.pos_includes.disabled = gui.corpus_type.value == 'text'
        gui.lemmatize.disabled = gui.corpus_type.value == 'text'

    gui.button.on_click(on_button_clicked)
    gui.corpus_type.observe(corpus_type_changed, 'value')

    return widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            gui.input_filename,
                            gui.corpus_type,
                            gui.output_tag,
                            gui.pos_includes,
                            gui.context_width,
                            gui.count_threshold,
                        ]
                    ),
                    widgets.VBox([gui.lemmatize, gui.to_lowercase, gui.remove_stopwords, gui.button]),
                ]
            ),
            gui.output,
        ]
    )
