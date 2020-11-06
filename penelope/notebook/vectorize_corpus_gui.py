import glob
import os
import types
from typing import Callable

import ipywidgets as widgets
from penelope.corpus.readers import AnnotationOpts
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
        input_filename=widgets.Dropdown(description='Corpus', options=corpus_filenames, value=None, layout=lw('350px')),
        corpus_type=widgets.Dropdown(
            description='Type', options=['text', 'sparv4-csv'], value='sparv4-csv', layout=lw('350px')
        ),
        output_tag=widgets.Text(
            value='',
            placeholder='Tag to prepend filenames',
            description='Output tag',
            disabled=False,
            layout=lw('350px'),
        ),
        pos_includes=widgets.SelectMultiple(
            options=SUC_PoS_tag_groups,
            value=[SUC_PoS_tag_groups['Noun'], SUC_PoS_tag_groups['Verb']],
            rows=8,
            description='PoS',
            disabled=False,
            layout=lw('350px'),
        ),
        count_threshold=widgets.IntSlider(
            description='Min Count', min=1, max=1000, step=1, value=1, layout=lw('350px')
        ),
        filename_fields=widgets.Text(
            value=r"year:prot\_(\d{4}).*",
            placeholder='Fields to extract from filename (regex)',
            description='Fields',
            disabled=False,
            layout=lw('350px'),
        ),
        lemmatize=widgets.ToggleButton(value=True, description='Lemmatize', icon='check', layout=lw('140px')),
        to_lowercase=widgets.ToggleButton(value=True, description='To Lower', icon='check', layout=lw('140px')),
        remove_stopwords=widgets.ToggleButton(value=True, description='No Stopwords', icon='check', layout=lw('140px')),
        only_alphabetic=widgets.ToggleButton(value=False, description='Only Alpha', icon='', layout=lw('140px')),
        only_any_alphanumeric=widgets.ToggleButton(
            value=False, description='Only Alphanum', icon='', layout=lw('140px')
        ),
        extra_stopwords_label=widgets.Label("Extra stopwords"),
        extra_stopwords=widgets.Textarea(
            value='örn',
            placeholder='Enter extra stop words',
            description='',
            disabled=False,
            layout=widgets.Layout(width='350px', height='100px'),
        ),
        button=widgets.Button(
            description='Vectorize!',
            button_style='Success',
            layout=lw('140px'),
        ),
        output=widgets.Output(),
    )

    def on_button_clicked(_):

        try:
            with gui.output as out:

                if gui.input_filename.value is None:
                    return

                output_folder = data_folder

                input_filename = os.path.join(data_folder, gui.input_filename.value)

                if not os.path.isfile(input_filename):
                    raise FileNotFoundError(input_filename)

                output_tag = gui.output_tag.value.strip()
                if output_tag == "":
                    print("please specify a unique string that will be prepended to output file")
                    return

                gui.button.disabled = True
                extra_stopwords = None
                if gui.extra_stopwords.value.strip() != '':
                    _words = [x for x in map(str.strip, gui.extra_stopwords.value.strip().split()) if x != '']
                    if len(_words) > 0:
                        extra_stopwords = _words

                tokens_transform_opts = TokensTransformOpts(
                    remove_stopwords=gui.remove_stopwords.value,
                    to_lower=gui.to_lowercase.value,
                    only_alphabetic=gui.only_alphabetic.value,
                    only_any_alphanumeric=gui.only_any_alphanumeric.value,
                    extra_stopwords=extra_stopwords,
                )
                out.clear_output()

                if gui.corpus_type.value == 'sparv4-csv':
                    annotation_opts = AnnotationOpts(
                        pos_includes=f"|{'|'.join(flatten(gui.pos_includes.value))}|",
                        pos_excludes="|MAD|MID|PAD|",
                        lemmatize=gui.lemmatize.value,
                    )
                    vectorize_sparv_csv_corpus_workflow(
                        input_filename=input_filename,
                        output_folder=output_folder,
                        output_tag=output_tag,
                        filename_field=gui.filename_fields.value,
                        count_threshold=gui.count_threshold.value,
                        annotation_opts=annotation_opts,
                        tokens_transform_opts=tokens_transform_opts,
                    )

                elif gui.corpus_type.value == 'text':

                    vectorize_tokenized_corpus_workflow(
                        input_filename=input_filename,
                        output_folder=output_folder,
                        output_tag=output_tag,
                        filename_field=gui.filename_fields.value,
                        count_threshold=gui.count_threshold.value,
                        tokens_transform_opts=tokens_transform_opts,
                    )

                if generated_callback is not None:
                    generated_callback(gui.output_tag.value, gui.output)

                gui.button.disabled = False

        except Exception as ex:
            print(ex)

    def corpus_type_changed(*_):
        gui.pos_includes.disabled = gui.corpus_type.value == 'text'
        gui.lemmatize.disabled = gui.corpus_type.value == 'text'

    def toggle_state_changed(event):
        with gui.output:
            try:
                event['owner'].icon = 'check' if event['new'] else ''
            except Exception as ex:
                print(event)
                print(ex)

    def remove_stopwords_state_changed(*_):
        gui.extra_stopwords.disabled = not gui.remove_stopwords.value

    gui.button.on_click(on_button_clicked)
    gui.corpus_type.observe(corpus_type_changed, 'value')
    gui.lemmatize.observe(toggle_state_changed, 'value')
    gui.to_lowercase.observe(toggle_state_changed, 'value')
    gui.remove_stopwords.observe(toggle_state_changed, 'value')
    gui.remove_stopwords.observe(remove_stopwords_state_changed, 'value')
    gui.only_alphabetic.observe(toggle_state_changed, 'value')
    gui.only_any_alphanumeric.observe(toggle_state_changed, 'value')

    return widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.VBox(
                        [
                            gui.input_filename,
                            gui.corpus_type,
                            gui.output_tag,
                            gui.filename_fields,
                            gui.pos_includes,
                            gui.count_threshold,
                        ]
                    ),
                    widgets.VBox(
                        [
                            widgets.VBox(
                                [
                                    gui.extra_stopwords_label,
                                    gui.extra_stopwords,
                                ]
                            ),
                            widgets.HBox(
                                [
                                    widgets.VBox(
                                        [
                                            gui.lemmatize,
                                            gui.to_lowercase,
                                            gui.remove_stopwords,
                                        ]
                                    ),
                                    widgets.VBox(
                                        [
                                            gui.only_alphabetic,
                                            gui.only_any_alphanumeric,
                                            gui.button,
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
            gui.output,
        ]
    )
