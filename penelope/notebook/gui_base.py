import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

import ipywidgets as widgets
import penelope.notebook.utility as notebook_utility
import penelope.utility as utility
from penelope.corpus import TokensTransformOpts, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.pipeline import CorpusConfig, CorpusType
from penelope.utility import PropertyValueMaskingOpts, better_flatten, default_data_folder, get_logger

from . import interface

logger = get_logger('penelope')

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes

default_layout = widgets.Layout(width='200px')
button_layout = widgets.Layout(width='140px')

# view: widgets.Output = widgets.Output()


@dataclass
class BaseGUI:

    default_corpus_path: str = None
    default_corpus_filename: str = field(default='')
    default_data_folder: str = None

    _config: CorpusConfig = None

    _corpus_filename: notebook_utility.FileChooserExt2 = None
    _target_folder: notebook_utility.FileChooserExt2 = None

    _corpus_tag: widgets.Text = widgets.Text(
        value='',
        placeholder='Tag to prepend output files',
        description='',
        disabled=False,
        layout=default_layout,
    )
    _pos_includes: widgets.SelectMultiple = widgets.SelectMultiple(
        options=[],
        value=[],
        rows=8,
        description='',
        disabled=False,
        layout=widgets.Layout(width='160px'),
    )
    _pos_paddings: widgets.SelectMultiple = widgets.SelectMultiple(
        options=[],
        value=[],
        rows=8,
        description='',
        disabled=False,
        layout=widgets.Layout(width='160px'),
    )
    _pos_excludes: widgets.SelectMultiple = widgets.SelectMultiple(
        options=[],
        value=[],
        rows=8,
        description='',
        disabled=False,
        layout=widgets.Layout(width='160px'),
    )
    _filename_fields: widgets.Text = widgets.Text(
        value="",
        placeholder='Fields to extract from filename (regex)',
        description='',
        disabled=True,
        layout=default_layout,
    )
    _create_subfolder: widgets.ToggleButton = widgets.ToggleButton(
        value=True, description='Create folder', icon='check', layout=button_layout
    )
    _lemmatize: widgets.ToggleButton = widgets.ToggleButton(
        value=True, description='Lemmatize', icon='check', layout=button_layout
    )
    _to_lowercase: widgets.ToggleButton = widgets.ToggleButton(
        value=True, description='To lower', icon='check', layout=button_layout
    )
    _remove_stopwords: widgets.ToggleButton = widgets.ToggleButton(
        value=False, description='No stopwords', icon='check', layout=button_layout
    )
    _only_alphabetic: widgets.ToggleButton = widgets.ToggleButton(
        value=False, description='Only alphabetic', icon='', layout=button_layout
    )
    _only_any_alphanumeric: widgets.ToggleButton = widgets.ToggleButton(
        value=False, description='Only alphanumeric', icon='', layout=button_layout
    )
    _extra_stopwords: widgets.Textarea = widgets.Textarea(
        value='örn',
        placeholder='Enter extra stop words',
        description='',
        disabled=False,
        rows=8,
        layout=widgets.Layout(width='100px'),
    )
    _count_threshold: widgets.IntSlider = widgets.IntSlider(
        description='', min=1, max=1000, step=1, value=10, layout=default_layout
    )
    _use_pos_groupings: widgets.ToggleButton = widgets.ToggleButton(
        value=True, description='PoS groups', icon='', layout=button_layout
    )
    # FIXME: #63 Append PoS togge button missing in GUI
    _append_pos_tag: widgets.ToggleButton = widgets.ToggleButton(
        value=False, description='Append PoS', icon='', layout=button_layout
    )
    _phrases = widgets.Text(
        value='',
        placeholder='Enter phrases, use semicoloon (;) as phrase delimiter',
        description='',
        disabled=False,
        layout=widgets.Layout(width='480px'),
    )

    _vectorize_button: widgets.Button = widgets.Button(
        description='Compute!',
        button_style='Success',
        layout=button_layout,
    )
    extra_placeholder: widgets.HBox = widgets.HBox()
    buttons_placeholder: widgets.VBox = widgets.VBox()

    _corpus_type: widgets.Dropdown = widgets.Dropdown(
        description='',
        options={
            'Text': CorpusType.Text,
            'Pipeline': CorpusType.Pipeline,
            'Sparv4-CSV': CorpusType.SparvCSV,
        },
        value=CorpusType.Pipeline,
        layout=default_layout,
    )

    compute_callback: Callable[[interface.ComputeOpts, CorpusConfig], Any] = None
    done_callback: Callable[[Any, interface.ComputeOpts], None] = None

    def layout(self, hide_input=False, hide_output=False) -> widgets.VBox:

        return widgets.VBox(
            (
                []
                if hide_input
                else [
                    widgets.HBox(
                        [widgets.VBox([widgets.HTML("<b>Corpus type</b>"), self._corpus_type]), self._corpus_filename]
                    )
                ]
            )
            + (
                []
                if hide_output
                else [
                    widgets.HBox(
                        [widgets.VBox([widgets.HTML("<b>Output tag</b>"), self._corpus_tag]), self._target_folder]
                    ),
                ]
            )
            + [
                self.extra_placeholder,
                widgets.HBox(
                    [
                        widgets.VBox([widgets.HTML("<b>Target PoS</b>"), self._pos_includes]),
                        widgets.VBox([widgets.HTML("<b>Padding PoS</b>"), self._pos_paddings]),
                        widgets.VBox([widgets.HTML("<b>Exclude PoS</b>"), self._pos_excludes]),
                        widgets.VBox([widgets.HTML("<b>Extra stopwords</b>"), self._extra_stopwords]),
                    ]
                ),
                widgets.HBox([widgets.VBox([widgets.HTML("<b>Phrases</b>"), self._phrases])]),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.VBox([widgets.HTML("<b>Filename fields</b>"), self._filename_fields]),
                                widgets.VBox([widgets.HTML("<b>Frequency threshold</b>"), self._count_threshold]),
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.HBox(
                                    [
                                        widgets.VBox(
                                            [
                                                self._use_pos_groupings,
                                                self._lemmatize,
                                                self._to_lowercase,
                                                self._remove_stopwords,
                                            ]
                                        ),
                                        self.buttons_placeholder,
                                        widgets.VBox(
                                            [
                                                self._append_pos_tag,
                                                self._only_alphabetic,
                                                self._only_any_alphanumeric,
                                                self._create_subfolder,
                                            ]
                                        ),
                                        widgets.VBox(
                                            [
                                                widgets.HTML("&nbsp;"),
                                                widgets.HTML("&nbsp;"),
                                                widgets.HTML("&nbsp;"),
                                                self._vectorize_button,
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
    def _compute_handler(self, *_):

        if self.compute_callback is None:
            raise ValueError("fatal: cannot compute (callback is not specified)")

        self._vectorize_button.disabled = True
        try:

            result: Any = self.compute_callback(self.compute_opts, self.corpus_config)

            if self.done_callback is not None:
                self.done_callback(result, self.compute_opts)

        except (ValueError, FileNotFoundError) as ex:
            print(ex)
        except Exception as ex:
            logger.info(ex)
            raise
        finally:
            self._vectorize_button.disabled = False

    def _corpus_type_changed(self, *_):
        self._pos_includes.disabled = self._corpus_type.value == 'text'
        self._pos_paddings.disabled = self._corpus_type.value == 'text'
        self._lemmatize.disabled = self._corpus_type.value == 'text'

    # @view.capture(clear_output=True)
    def _toggle_state_changed(self, event):
        try:
            event['owner'].icon = 'check' if event['new'] else ''
        except Exception as ex:
            logger.exception(ex)

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
            path=self.default_corpus_path or default_data_folder(),
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
            path=self.default_data_folder or default_data_folder(),
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
        self._vectorize_button.on_click(self._compute_handler)
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
            self._pos_includes.value = [pos_schema.groups['Noun'], pos_schema.groups['Verb']]
            self._pos_paddings.value = []
            self._pos_excludes.value = [pos_schema.groups['Delimiter']]
        else:
            self._pos_includes.value = pos_schema.groups['Noun'] + pos_schema.groups['Verb']
            self._pos_paddings.value = []
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
    def tokens_transform_opts(self) -> TokensTransformOpts:

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
    def extract_tagged_tokens_opts(self) -> ExtractTaggedTokensOpts:
        return ExtractTaggedTokensOpts(
            pos_includes=f"|{'|'.join(better_flatten(self._pos_includes.value))}|",
            pos_paddings=f"|{'|'.join(better_flatten(self._pos_paddings.value))}|",
            pos_excludes=f"|{'|'.join(better_flatten(self._pos_excludes.value))}|",
            lemmatize=self._lemmatize.value,
            passthrough_tokens=list(),
            phrases=self.phrases,
            to_lowercase=self._to_lowercase.value,
            append_pos=self._append_pos_tag.value,
        )

    @property
    def tagged_tokens_filter_opts(self) -> PropertyValueMaskingOpts:
        # FIXME #48 Check if _only_alphabetic is valid for Stanza & Sparv (or ignored)
        return PropertyValueMaskingOpts(
            is_alpha=self._only_alphabetic.value, is_punct=False, is_digit=None, is_stop=None
        )

    @property
    def vectorize_opts(self) -> VectorizeOpts:
        return VectorizeOpts(already_tokenized=True, lowercase=False, max_df=1.0, min_df=1, verbose=False)

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
    def count_threshold(self) -> int:
        return self._count_threshold.value

    @property
    def create_subfolder(self) -> bool:
        return self._create_subfolder.value

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
            corpus_filename=self.corpus_filename,
            target_folder=self.target_folder,
            corpus_tag=self.corpus_tag,
            tokens_transform_opts=self.tokens_transform_opts,
            text_reader_opts=self.corpus_config.text_reader_opts,
            extract_tagged_tokens_opts=self.extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=self.tagged_tokens_filter_opts,
            count_threshold=self.count_threshold,
            create_subfolder=self.create_subfolder,
            vectorize_opts=self.vectorize_opts,
            persist=True,
            context_opts=None,
        )
        return args
