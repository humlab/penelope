import glob
import os
import types
from typing import Callable

import ipywidgets as widgets
from penelope.co_occurrence.concept_co_occurrence import ConceptContextOpts
from penelope.corpus.readers import AnnotationOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.utility import flatten, replace_extension, strip_paths
from penelope.utility.tags import SUC_PoS_tag_groups
from penelope.workflows import concept_co_occurrence_workflow

# logger = getLogger('corpus_text_analysis')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


def display_gui(data_folder: str, corpus_pattern: str, generated_callback: Callable):
    lw = lambda w: widgets.Layout(width=w)
    corpus_filenames = list(map(strip_paths, sorted(glob.glob(os.path.join(data_folder, corpus_pattern)))))

    # Hard coded for now, must be changed!!!!
    filename_field = {"year": r"prot\_(\d{4}).*"}
    partition_keys = "year"

    gui = types.SimpleNamespace(
        input_filename=widgets.Dropdown(description='Corpus', options=corpus_filenames, value=None, layout=lw('400px')),
        output_filename=widgets.Text(
            value='',
            placeholder='Enter a filename without path or extension',
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
        filename_fields=widgets.Text(
            value=r"year:prot\_(\d{4}).*",
            placeholder='Fields to extract from filename (regex)',
            description='Fields',
            disabled=True,
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

        input_filename = os.path.join(data_folder, gui.input_filename.value)
        output_filename = replace_extension(gui.output_filename.value.strip(), 'csv.zip')
        if not os.path.isabs(output_filename):
            output_filename = os.path.join(data_folder, output_filename)

        if not os.path.isfile(input_filename):
            raise FileNotFoundError(input_filename)

        with gui.output:

            gui.button.disabled = True

            concept = set(map(str.strip, gui.concept.value.split(',')))

            tokens_transform_opts = TokensTransformOpts(
                to_lower=gui.to_lowercase.value,
                to_upper=False,
                remove_stopwords=gui.remove_stopwords.value,
                extra_stopwords=None,
                language='swedish' if gui.remove_stopwords.value else None,
                keep_numerals=False,
                keep_symbols=False,
                only_alphabetic=False,
                only_any_alphanumeric=True,
            )
            annotation_opts = AnnotationOpts(
                pos_includes=f"|{'|'.join(flatten(gui.pos_includes.value))}|",
                pos_excludes="|MAD|MID|PAD|",
                lemmatize=gui.lemmatize.value,
            )

            concept_opts = ConceptContextOpts(
                concept=concept, context_width=gui.context_width.value, ignore_concept=gui.no_concept.value
            )
            count_threshold = None if gui.count_threshold.value < 2 else gui.count_threshold.value

            concept_co_occurrence_workflow(
                input_filename=input_filename,
                output_filename=output_filename,
                concept_opts=concept_opts,
                count_threshold=count_threshold,
                partition_keys=partition_keys,
                filename_field=filename_field,
                annotation_opts=annotation_opts,
                tokens_transform_opts=tokens_transform_opts,
                store_vectorized=True,
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
