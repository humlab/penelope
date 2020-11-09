import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ipyfilechooser
import ipywidgets as widgets
from penelope.co_occurrence.concept_co_occurrence import ConceptContextOpts
from penelope.corpus.readers import AnnotationOpts
from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.utility import filename_whitelist, flatten, getLogger, replace_extension
from penelope.utility.tags import SUC_PoS_tag_groups
from penelope.workflows import concept_co_occurrence_workflow

logger = getLogger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes


def _label(x):
    return widgets.HTML(f'<b>{x}</b>')


def _layout(width, **kwargs):
    return widgets.Layout(width=width, **kwargs)


@dataclass
class GUI:

    col_layout = _layout('400px')
    button_layout = _layout('120px')
    input_filename_chooser = ipyfilechooser.FileChooser(
        path='/data/westac',
        filter_pattern='*.zip',
        title='<b>Corpus file (Sparv v4 annotated, CSV exported)</b>',
        show_hidden=False,
        select_default=False,
        use_dir_icons=True,
        show_only_dirs=False,
    )
    output_folder_chooser = ipyfilechooser.FileChooser(
        path=str(Path.home()),
        title='<b>Output folder</b>',
        show_hidden=False,
        select_default=True,
        use_dir_icons=True,
        show_only_dirs=True,
    )
    output_tag = widgets.Text(
        value='',
        placeholder='Tag will be prepended to result filename',
        description='',
        disabled=False,
        layout=col_layout,
    )
    concept = widgets.Text(
        value='',
        placeholder='Use comma (,) as word delimiter',
        description='',
        disabled=False,
        layout=col_layout,
    )
    pos_includes = widgets.SelectMultiple(
        options=SUC_PoS_tag_groups,
        value=[SUC_PoS_tag_groups['Noun'], SUC_PoS_tag_groups['Verb']],
        rows=8,
        description='',
        disabled=False,
        layout=_layout('400px'),
    )
    filename_fields = widgets.Text(
        value=r"year:prot\_(\d{4}).*",
        placeholder='Fields to extract from filename (regex)',
        description='',
        disabled=True,
        layout=col_layout,
    )
    context_width = widgets.IntSlider(description='', min=1, max=20, step=1, value=2, layout=col_layout)
    count_threshold = widgets.IntSlider(description='Min Count', min=1, max=1000, step=1, value=1, layout=col_layout)
    no_concept = widgets.ToggleButton(value=True, description='No Concept', icon='check', layout=button_layout)
    lemmatize = widgets.ToggleButton(value=True, description='Lemmatize', icon='check', layout=button_layout)
    to_lowercase = widgets.ToggleButton(value=True, description='To Lower', icon='check', layout=button_layout)
    remove_stopwords = widgets.ToggleButton(value=True, description='No Stopwords', icon='check', layout=button_layout)
    create_subfolder = widgets.ToggleButton(value=True, description='Create folder', icon='check', layout=button_layout)

    button = widgets.Button(
        description='Compute',
        button_style='Success',
        layout=button_layout,
    )
    output = widgets.Output()

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                self.input_filename_chooser,
                                widgets.VBox([_label("Concept tokens"), self.concept]),
                                widgets.VBox([_label("PoS group filter"), self.pos_includes]),
                                self.count_threshold,
                            ]
                        ),
                        widgets.VBox(
                            [
                                self.output_folder_chooser,
                                widgets.VBox([_label("Output tag"), self.output_tag]),
                                widgets.VBox([_label("Filename metadata fields"), self.filename_fields]),
                                widgets.VBox([_label("Width (max distance to concept)"), self.context_width]),
                                widgets.HBox(
                                    [
                                        self.no_concept,
                                        self.lemmatize,
                                        self.to_lowercase,
                                    ]
                                ),
                                widgets.HBox(
                                    [
                                        self.remove_stopwords,
                                        self.create_subfolder,
                                        self.button,
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                self.output,
            ]
        )


def display_gui(
    *, data_folder: str, corpus_pattern: str, generated_callback: Callable[[widgets.Output, str, str, str], None]
):

    # Hard coded for now, must be changed!!!!
    filename_field = {"year": r"prot\_(\d{4}).*"}
    partition_keys = "year"

    gui = GUI()
    gui.input_filename_chooser.path = data_folder
    gui.input_filename_chooser.filter_pattern = corpus_pattern

    def on_button_clicked(_):

        try:
            if gui.input_filename_chooser.selected is None:
                raise ValueError("Please select a corpus file")

            if gui.output_folder_chooser.selected == "":
                raise ValueError("Please choose where to store result")

            output_tag = filename_whitelist(gui.output_tag.value.strip())
            if output_tag == "":
                raise ValueError("Please specify output tag")

            if not os.access(gui.output_folder_chooser.selected, os.W_OK):
                raise PermissionError("You lack write permission to folder")

            if not os.path.isfile(gui.input_filename_chooser.selected):
                raise FileNotFoundError(gui.input_filename_chooser.selected)

            output_folder = gui.output_folder_chooser.selected
            if gui.create_subfolder.value:
                output_folder = os.path.join(output_folder, output_tag)
                os.makedirs(output_folder, exist_ok=True)

            output_filename = os.path.join(
                output_folder,
                replace_extension(output_tag, '.coo_concept_context.csv.zip'),
            )

            gui.output.clear_output()

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

                concept_co_occurrences = concept_co_occurrence_workflow(
                    input_filename=gui.input_filename_chooser.selected,
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
                    generated_callback(
                        output=gui.output,
                        corpus_folder=output_folder,
                        corpus_tag=gui.output_tag.value,
                        concept_co_occurrences=concept_co_occurrences,
                        concept_co_occurrences_filename=output_filename,
                    )

        except (
            ValueError,
            FileNotFoundError,
            PermissionError,
        ) as ex:
            with gui.output:
                logger.error(ex)
        except Exception as ex:
            with gui.output:
                logger.error(ex)

        finally:
            gui.button.disabled = False

    gui.button.on_click(on_button_clicked)

    return gui.layout()
