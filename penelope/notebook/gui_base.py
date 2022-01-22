import contextlib
import os
from typing import Any, Callable, Dict, List

from ipywidgets import (
    HTML,
    Button,
    Dropdown,
    HBox,
    IntSlider,
    Layout,
    SelectMultiple,
    Text,
    Textarea,
    ToggleButton,
    VBox,
)
from loguru import logger

import penelope.utility as utility
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig, CorpusType
from penelope.utility import better_flatten
from penelope.utility import default_data_folder as home_data_folder
from penelope.workflows import interface

from . import utility as notebook_utility

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes, too-many-public-methods

default_layout = Layout(width='200px')
button_layout = Layout(width='140px')

# view:Output = Output()


class BaseGUI:
    def __init__(
        self, default_corpus_path: str = None, default_corpus_filename: str = '', default_data_folder: str = None
    ):
        self.default_corpus_path: str = default_corpus_path
        self.default_corpus_filename: str = os.path.basename(default_corpus_filename or '')
        self.default_data_folder: str = default_data_folder

        self._config: CorpusConfig = None

        self._corpus_filename: notebook_utility.FileChooserExt2 = None
        self._target_folder: notebook_utility.FileChooserExt2 = None

        self._corpus_tag: Text = Text(
            value='',
            placeholder='Tag to prepend output files',
            description='',
            disabled=False,
            layout=default_layout,
        )
        self._pos_includes: SelectMultiple = SelectMultiple(
            options=[],
            value=[],
            rows=8,
            description='',
            disabled=False,
            layout=Layout(width='160px'),
        )
        self._pos_paddings: SelectMultiple = SelectMultiple(
            options=[],
            value=[],
            rows=8,
            description='',
            disabled=False,
            layout=Layout(width='160px'),
        )
        self._pos_excludes: SelectMultiple = SelectMultiple(
            options=[],
            value=[],
            rows=8,
            description='',
            disabled=False,
            layout=Layout(width='160px'),
        )
        self._filename_fields: Text = Text(
            value="",
            placeholder='Fields to extract from filename (regex)',
            description='',
            disabled=True,
            layout=default_layout,
        )
        self._create_subfolder: ToggleButton = ToggleButton(
            value=True, description='Create folder', icon='check', layout=button_layout
        )
        self._ignore_checkpoints: ToggleButton = ToggleButton(
            value=False, description='Force', icon='', layout=button_layout
        )
        self._lemmatize: ToggleButton = ToggleButton(
            value=True, description='Lemmatize', icon='check', layout=button_layout
        )
        self._to_lowercase: ToggleButton = ToggleButton(
            value=True, description='To lower', icon='check', layout=button_layout
        )
        self._remove_stopwords: ToggleButton = ToggleButton(
            value=False, description='No stopwords', icon='', layout=button_layout
        )
        self._only_alphabetic: ToggleButton = ToggleButton(
            value=False, description='Only alpha', icon='', layout=button_layout
        )
        self._only_any_alphanumeric: ToggleButton = ToggleButton(
            value=False, description='Only alphanum', icon='', layout=button_layout
        )
        self._extra_stopwords: Textarea = Textarea(
            value='',
            placeholder='Enter extra stop words',
            description='',
            disabled=False,
            rows=8,
            layout=Layout(width='140px'),
        )
        self._tf_threshold: IntSlider = IntSlider(
            description='', min=1, max=1000, step=1, value=10, layout=default_layout
        )
        self._tf_threshold_mask: ToggleButton = ToggleButton(
            value=False, description='Mask low-TF', icon='', layout=button_layout
        )
        self._use_pos_groupings: ToggleButton = ToggleButton(
            value=True, description='PoS groups', icon='check', layout=button_layout
        )

        self._append_pos_tag: ToggleButton = ToggleButton(
            value=False, description='Append PoS', icon='', layout=button_layout
        )
        self._phrases = Text(
            value='',
            placeholder='Enter phrases, use semicolon (;) as phrase delimiter',
            description='',
            disabled=False,
            layout=Layout(width='480px'),
        )

        self._compute_button: Button = Button(
            description='Compute!',
            button_style='Success',
            layout=button_layout,
        )
        self._cli_button: Button = Button(description='CLI', button_style='Success', layout=button_layout)
        self.extra_placeholder: HBox = HBox()
        self.buttons_placeholder: VBox = VBox()

        self._corpus_type: Dropdown = Dropdown(
            description='',
            options={
                'Text': CorpusType.Text,
                'Pipeline': CorpusType.Pipeline,
                'Sparv4-CSV': CorpusType.SparvCSV,
            },
            value=CorpusType.Pipeline,
            layout=default_layout,
        )

        self.compute_callback: Callable[[interface.ComputeOpts, CorpusConfig], Any] = None
        self.done_callback: Callable[[Any, interface.ComputeOpts], None] = None

    def layout(self, hide_input=False, hide_output=False) -> VBox:

        return VBox(
            (
                []
                if hide_input
                else [HBox([VBox([HTML("<b>Corpus type</b>"), self._corpus_type]), self._corpus_filename])]
            )
            + (
                []
                if hide_output
                else [
                    HBox([VBox([HTML("<b>Output tag</b>"), self._corpus_tag]), self._target_folder]),
                ]
            )
            + [
                self.extra_placeholder,
                HBox(
                    [
                        VBox([HTML("<b>Target PoS</b>"), self._pos_includes]),
                        VBox([HTML("<b>Padding PoS</b>"), self._pos_paddings]),
                        VBox([HTML("<b>Exclude PoS</b>"), self._pos_excludes]),
                        VBox([HTML("<b>Extra stopwords</b>"), self._extra_stopwords]),
                    ]
                ),
                HBox([VBox([HTML("<b>Phrases</b>"), self._phrases])]),
                HBox(
                    [
                        VBox(
                            [
                                VBox([HTML("<b>Filename fields</b>"), self._filename_fields]),
                                VBox([HTML("<b>Frequency threshold</b>"), self._tf_threshold]),
                            ]
                        ),
                        VBox(
                            [
                                HBox(
                                    [
                                        VBox(
                                            [
                                                self._use_pos_groupings,
                                                self._lemmatize,
                                                self._to_lowercase,
                                                self._tf_threshold_mask,
                                            ]
                                        ),
                                        self.buttons_placeholder,
                                        VBox(
                                            [
                                                self._append_pos_tag,
                                                self._only_alphabetic,
                                                self._only_any_alphanumeric,
                                                self._remove_stopwords,
                                            ]
                                        ),
                                        VBox(
                                            [
                                                self._ignore_checkpoints,
                                                self._create_subfolder,
                                                self._cli_button,
                                                self._compute_button,
                                            ]
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
                # view,
            ]
        )

    # @view.capture(clear_output=True)
    def _compute_handler(self, sender: Button, *_):

        if self.compute_callback is None:
            raise ValueError("fatal: cannot compute (callback is not specified)")

        self._compute_button.disabled = True
        self._cli_button.disabled = True
        try:

            opts: interface.ComputeOpts = self.compute_opts
            opts.dry_run = sender.description == "CLI"
            result: Any = self.compute_callback(opts, self.corpus_config)

            if result is not None and self.done_callback is not None:
                self.done_callback(result, self.compute_opts)

        except (ValueError, FileNotFoundError) as ex:
            print(ex)
        except Exception as ex:
            logger.info(ex)
            raise
        finally:
            self._cli_button.disabled = False
            self._compute_button.disabled = False

    def _corpus_type_changed(self, *_):
        self._pos_includes.disabled = self._corpus_type.value == 'text'
        self._pos_paddings.disabled = self._corpus_type.value == 'text'
        self._lemmatize.disabled = self._corpus_type.value == 'text'

    # @view.capture(clear_output=True)
    def _toggle_state_changed(self, event):
        with contextlib.suppress(Exception):
            event['owner'].icon = 'check' if event['new'] else ''

    def _remove_stopwords_state_changed(self, *_):
        self._extra_stopwords.disabled = not self._remove_stopwords.value

    def setup(
        self,
        *,
        config: CorpusConfig,
        compute_callback: Callable[[interface.ComputeOpts, CorpusConfig], Any],
        done_callback: Callable[[Any, interface.ComputeOpts], None],
    ) -> "BaseGUI":

        self._corpus_filename = notebook_utility.FileChooserExt2(
            path=self.default_corpus_path or home_data_folder(),
            filename=self.default_corpus_filename,
            filter_pattern=config.corpus_pattern,
            title='<b>Source corpus file</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False,
        )
        self._corpus_filename.refresh()

        self._target_folder = notebook_utility.FileChooserExt2(
            path=self.default_data_folder or home_data_folder(),
            title='<b>Output folder</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=True,
        )
        self._target_folder.refresh()

        self._corpus_type.value = config.corpus_type
        self._corpus_type.disabled = True
        self._corpus_filename.filter_pattern = config.corpus_pattern
        self._filename_fields.value = ';'.join(config.text_reader_opts.filename_fields)
        self._filename_fields.disabled = True

        self._corpus_type.observe(self._corpus_type_changed, 'value')
        self._lemmatize.observe(self._toggle_state_changed, 'value')
        self._to_lowercase.observe(self._toggle_state_changed, 'value')
        self._remove_stopwords.observe(self._toggle_state_changed, 'value')
        self._remove_stopwords.observe(self._remove_stopwords_state_changed, 'value')
        self._only_alphabetic.observe(self._toggle_state_changed, 'value')
        self._only_any_alphanumeric.observe(self._toggle_state_changed, 'value')
        self._use_pos_groupings.observe(self._toggle_state_changed, 'value')
        self._tf_threshold_mask.observe(self._toggle_state_changed, 'value')
        self._create_subfolder.observe(self._toggle_state_changed, 'value')
        self._ignore_checkpoints.observe(self._toggle_state_changed, 'value')
        self._append_pos_tag.observe(self._toggle_state_changed, 'value')

        self._compute_button.on_click(self._compute_handler)
        self._cli_button.on_click(self._compute_handler)
        self._use_pos_groupings.observe(self.update_pos_schema, 'value')
        self._pos_includes.observe(self.pos_select_update, 'value')
        self._pos_paddings.observe(self.pos_select_update, 'value')
        self._pos_excludes.observe(self.pos_select_update, 'value')

        self.update_config(config)
        self.compute_callback = compute_callback
        self.done_callback = done_callback

        return self

    def update_config(self, __config: CorpusConfig) -> "BaseGUI":

        self._config = __config
        self.update_pos_schema({})
        return self

    def update_pos_schema(self, *_) -> "BaseGUI":

        pos_schema: utility.PoS_Tag_Scheme = self._config.pos_schema

        tags: Dict[str, str] = {f"{tag}/{description}": tag for tag, description in pos_schema.description.items()}

        for _pos_widget in [self._pos_includes, self._pos_paddings, self._pos_excludes]:
            _pos_widget.value = []
            _pos_widget.options = pos_schema.groups if self._use_pos_groupings.value else tags

        if self._use_pos_groupings.value:
            self._pos_paddings.value = [v for _, v in self._pos_paddings.options.items()]
            self._pos_includes.value = [pos_schema.groups['Noun'], pos_schema.groups['Verb']]
            self._pos_excludes.value = [pos_schema.groups['Delimiter']]
        else:
            self._pos_paddings.value = [v for _, v in self._pos_paddings.options.items()]
            self._pos_includes.value = pos_schema.groups['Noun'] + pos_schema.groups['Verb']
            self._pos_excludes.value = pos_schema.groups['Delimiter']

        return self

    def pos_select_update(self, event):
        with contextlib.suppress(Exception):

            if event['name'] != 'value' or len(event['new']) < len(event['old']):
                return

            for _pos_widget in [self._pos_includes, self._pos_paddings, self._pos_excludes]:
                if _pos_widget is event['owner']:
                    continue
                _pos_widget.unobserve(self.pos_select_update, 'value')
                _pos_widget.value = [x for x in _pos_widget.value if x not in event['new']]
                _pos_widget.observe(self.pos_select_update, 'value')

    @property
    def transform_opts(self) -> TokensTransformOpts:

        opts = TokensTransformOpts(
            keep_numerals=True,
            keep_symbols=True,
            language=self._config.language,
            max_len=None,
            min_len=1,
            only_alphabetic=self._only_alphabetic.value,
            only_any_alphanumeric=self._only_any_alphanumeric.value,
            remove_accents=False,
            remove_stopwords=self._remove_stopwords.value,
            stopwords=None,
            to_lower=self._to_lowercase.value,
            to_upper=False,
        )

        if self._extra_stopwords.value.strip() != '':
            _words = [x for x in map(str.strip, self._extra_stopwords.value.strip().split()) if x != '']
            if len(_words) > 0:
                opts.extra_stopwords = _words

        return opts

    @property
    def extract_opts(self) -> ExtractTaggedTokensOpts:

        # if ["PASSTHROUGH"] in self._pos_paddings.value:
        #    pos_paddings = pos_tags_to_str(corpus_config.pos_schema.all_types_except(pos_includes))
        #    logger.info(f"PoS paddings expanded to: {pos_paddings}")

        return ExtractTaggedTokensOpts(
            pos_includes=f"|{'|'.join(better_flatten(self._pos_includes.value))}|",
            pos_paddings=f"|{'|'.join(better_flatten(self._pos_paddings.value))}|",
            pos_excludes=f"|{'|'.join(better_flatten(self._pos_excludes.value))}|",
            lemmatize=self._lemmatize.value,
            passthrough_tokens=[],
            block_tokens=[],
            phrases=self.phrases,
            append_pos=self._append_pos_tag.value,
            global_tf_threshold=self.tf_threshold,
            global_tf_threshold_mask=self.tf_threshold_mask,
            **self.corpus_config.pipeline_payload.tagged_columns_names,
            filter_opts=self.filter_opts,
        )

    @property
    def filter_opts(self) -> dict:
        return dict(is_alpha=self._only_alphabetic.value, is_punct=False, is_digit=None, is_stop=None)

    @property
    def vectorize_opts(self) -> VectorizeOpts:
        # FIXME: Add UI elements for max_tokens
        return VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            max_df=1.0,
            min_df=1,
            min_tf=self.tf_threshold,
        )

    @property
    def corpus_type(self) -> CorpusType:
        return self._corpus_type.value

    @property
    def corpus_tag(self) -> str:
        return self._corpus_tag.value.strip()

    @property
    def target_folder(self) -> str:
        if self._create_subfolder:
            return os.path.join(self._target_folder.selected_path, self.corpus_tag)
        return self._target_folder.selected_path

    @property
    def corpus_folder(self) -> str:
        return self._corpus_filename.selected_path

    @property
    def corpus_filename(self) -> str:
        return self._corpus_filename.selected

    @property
    def tf_threshold(self) -> int:
        return self._tf_threshold.value

    @property
    def tf_threshold_mask(self) -> bool:
        return self._tf_threshold_mask.value

    @property
    def create_subfolder(self) -> bool:
        return self._create_subfolder.value

    @property
    def ignore_checkpoints(self) -> bool:
        return self._ignore_checkpoints.value

    @property
    def filename_fields(self) -> str:
        return self._filename_fields.value

    @property
    def corpus_config(self) -> CorpusConfig:
        return self._config

    @property
    def phrases(self) -> List[List[str]]:
        if not self._phrases.value.strip():
            return None
        _phrases = [phrase.split() for phrase in self._phrases.value.strip().split(';')]
        return _phrases

    @property
    def compute_opts(self) -> interface.ComputeOpts:
        args: interface.ComputeOpts = interface.ComputeOpts(
            corpus_type=self.corpus_type,
            corpus_source=self.corpus_filename,
            target_folder=self.target_folder,
            corpus_tag=self.corpus_tag,
            transform_opts=self.transform_opts,
            text_reader_opts=self.corpus_config.text_reader_opts,
            extract_opts=self.extract_opts,
            tf_threshold=self.tf_threshold,
            tf_threshold_mask=self.tf_threshold_mask,
            create_subfolder=self.create_subfolder,
            vectorize_opts=self.vectorize_opts,
            persist=True,
            context_opts=None,
            enable_checkpoint=True,
            force_checkpoint=self.ignore_checkpoints,
        )
        return args
