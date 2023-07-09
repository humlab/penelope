import contextlib
from os.path import basename, dirname, join
from typing import Any, Protocol

import ipywidgets as w
from loguru import logger

import penelope.utility as utility
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig, CorpusType
from penelope.utility import better_flatten
from penelope.utility import default_data_folder as home_data_folder
from penelope.workflows import interface

from . import utility as notebook_utility

# pylint: disable=attribute-defined-outside-init, too-many-instance-attributes, too-many-public-methods

default_layout = w.Layout(width='200px')
button_layout = w.Layout(width='140px')

# view:Output = Output()


class DoneCallback(Protocol):
    def __call__(
        self,
        result: str | Any,
        folder: str | None = None,
        sender: Any | None = None,
        opts: interface.ComputeOpts = None,
        **kwargs: str,
    ) -> None:
        ...


class ComputeCallback(Protocol):
    def __call__(self, opts: interface.ComputeOpts, config: CorpusConfig) -> Any:
        ...


class BaseGUI:
    def __init__(
        self, default_corpus_path: str = None, default_corpus_filename: str = '', default_data_folder: str = None
    ):
        self._config: CorpusConfig = None

        self._config_chooser: notebook_utility.FileChooserExt2 = notebook_utility.FileChooserExt2(
            path=default_corpus_path or home_data_folder(),
            # filename="config.yml",
            filter_pattern="*.yml",
            title='<b>Config file</b>',
            select_default=True,
            use_dir_icons=True,
        )
        self._config_chooser.refresh()

        self._corpus_filename: notebook_utility.FileChooserExt2 = notebook_utility.FileChooserExt2(
            path=default_corpus_path or home_data_folder(),
            filename=basename(default_corpus_filename or ''),
            filter_pattern="*.*",
            title='<b>Source corpus file</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=False,
        )
        self._corpus_filename.refresh()

        self._target_folder: notebook_utility.FileChooserExt2 = notebook_utility.FileChooserExt2(
            path=default_data_folder or home_data_folder(),
            title='<b>Output folder</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=True,
        )
        self._target_folder.refresh()

        self._corpus_tag: w.Text = w.Text(
            value='',
            placeholder='Tag to prepend output files',
            description='',
            disabled=False,
            layout=default_layout,
        )
        self._pos_includes: w.SelectMultiple = w.SelectMultiple(
            options=[],
            value=[],
            rows=8,
            description='',
            disabled=False,
            layout=w.Layout(width='160px'),
        )
        self._pos_paddings: w.SelectMultiple = w.SelectMultiple(
            options=[],
            value=[],
            rows=8,
            description='',
            disabled=False,
            layout=w.Layout(width='160px'),
        )
        self._pos_excludes: w.SelectMultiple = w.SelectMultiple(
            options=[],
            value=[],
            rows=8,
            description='',
            disabled=False,
            layout=w.Layout(width='160px'),
        )
        self._filename_fields: w.Text = w.Text(
            value="",
            placeholder='Fields to extract from filename (regex)',
            description='',
            disabled=True,
            layout=default_layout,
        )
        self._create_subfolder: w.ToggleButton = w.ToggleButton(
            value=True, description='Create folder', icon='check', layout=button_layout
        )
        self._ignore_checkpoints: w.ToggleButton = w.ToggleButton(
            value=False, description='Force', icon='', layout=button_layout
        )
        self._lemmatize: w.ToggleButton = w.ToggleButton(
            value=True, description='Lemmatize', icon='check', layout=button_layout
        )
        self._to_lowercase: w.ToggleButton = w.ToggleButton(
            value=True, description='To lower', icon='check', layout=button_layout
        )
        self._remove_stopwords: w.ToggleButton = w.ToggleButton(
            value=False, description='No stopwords', icon='', layout=button_layout
        )
        self._only_alphabetic: w.ToggleButton = w.ToggleButton(
            value=False, description='Only alpha', icon='', layout=button_layout
        )
        self._only_any_alphanumeric: w.ToggleButton = w.ToggleButton(
            value=False, description='Only alphanum', icon='', layout=button_layout
        )
        self._extra_stopwords: w.Textarea = w.Textarea(
            value='',
            placeholder='Enter extra stop words',
            description='',
            disabled=False,
            rows=8,
            layout=w.Layout(width='140px'),
        )
        self._tf_threshold: w.IntSlider = w.IntSlider(
            description='', min=1, max=1000, step=1, value=10, layout=default_layout
        )
        self._tf_threshold_mask: w.ToggleButton = w.ToggleButton(
            value=False, description='Mask low-TF', icon='', layout=button_layout
        )
        self._use_pos_groupings: w.ToggleButton = w.ToggleButton(
            value=True, description='PoS groups', icon='check', layout=button_layout
        )

        self._append_pos_tag: w.ToggleButton = w.ToggleButton(
            value=False, description='Append PoS', icon='', layout=button_layout
        )
        self._phrases = w.Text(
            value='',
            placeholder='Enter phrases, use semicolon (;) as phrase delimiter',
            description='',
            disabled=False,
            layout=w.Layout(width='480px'),
        )

        self._compute_button: w.Button = w.Button(
            description='Compute!',
            button_style='Success',
            layout=button_layout,
        )
        self._cli_button: w.Button = w.Button(description='CLI', button_style='Success', layout=button_layout)
        self.extra_placeholder: w.HBox = w.HBox()
        self.buttons_placeholder: w.VBox = w.VBox()

        self._corpus_type: w.Dropdown = w.Dropdown(
            description='',
            options={
                'Text': CorpusType.Text,
                'Pipeline': CorpusType.Pipeline,
                'Sparv4-CSV': CorpusType.SparvCSV,
            },
            value=CorpusType.Pipeline,
            layout=default_layout,
        )

        self._alert: w.HTML = w.HTML("&nbsp;", layout={'width': '95%'})

        self.compute_callback: ComputeCallback = None
        self.done_callback: DoneCallback = None

    def alert(self, msg: str):
        self._alert.value = f"<b>{msg}</b>"

    def warn(self, msg: str):
        self.alert(f"<span style='color: red'>{msg or '😐'}</span>")

    def info(self, msg: str) -> None:
        self.alert(f"<span style='color: green'>{msg or '😃'}</span>")

    def layout(self, hide_input: bool = False, hide_output: bool = False) -> w.VBox:
        return w.VBox(
            (
                []
                if hide_input or self._config_chooser is None
                else [
                    w.HBox(
                        [
                            w.HTML("&nbsp;<p><b>Please select config file.</b>", layout={'width': '200px'}),
                            self._config_chooser,
                        ]
                    )
                ]
            )
            + (
                []
                if hide_input
                else [w.HBox([w.VBox([w.HTML("<b>Corpus type</b>"), self._corpus_type]), self._corpus_filename])]
            )
            + (
                []
                if hide_output
                else [
                    w.HBox([w.VBox([w.HTML("<b>Output tag</b>"), self._corpus_tag]), self._target_folder]),
                ]
            )
            + [
                self.extra_placeholder,
                w.HBox(
                    [
                        w.VBox([w.HTML("<b>Target PoS</b>"), self._pos_includes]),
                        w.VBox([w.HTML("<b>Padding PoS</b>"), self._pos_paddings]),
                        w.VBox([w.HTML("<b>Exclude PoS</b>"), self._pos_excludes]),
                        w.VBox([w.HTML("<b>Extra stopwords</b>"), self._extra_stopwords]),
                    ]
                ),
                w.HBox([w.VBox([w.HTML("<b>Phrases</b>"), self._phrases])]),
                w.HBox(
                    [
                        w.VBox(
                            [
                                w.VBox([w.HTML("<b>Filename fields</b>"), self._filename_fields]),
                                w.VBox([w.HTML("<b>Frequency threshold</b>"), self._tf_threshold]),
                            ]
                        ),
                        w.VBox(
                            [
                                w.HBox(
                                    [
                                        w.VBox(
                                            [
                                                self._use_pos_groupings,
                                                self._lemmatize,
                                                self._to_lowercase,
                                                self._tf_threshold_mask,
                                            ]
                                        ),
                                        self.buttons_placeholder,
                                        w.VBox(
                                            [
                                                self._append_pos_tag,
                                                self._only_alphabetic,
                                                self._only_any_alphanumeric,
                                                self._remove_stopwords,
                                            ]
                                        ),
                                        w.VBox(
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
                self._alert,
            ]
        )

    def _compute_handler(self, sender: w.Button, *_):
        if self.compute_callback is None:
            raise ValueError("fatal: cannot compute (callback is not specified)")

        self._compute_button.disabled = True
        self._cli_button.disabled = True
        try:
            opts: interface.ComputeOpts = self.compute_opts
            opts.dry_run = sender.description == "CLI"
            result: Any = self.compute_callback(opts=opts, config=self.corpus_config)

            if result is not None and self.done_callback is not None:
                self.done_callback(
                    result=result,
                    folder=dirname(opts.corpus_source),
                    opts=opts,
                    sender=self,
                    config=self.corpus_config,
                )

        except (ValueError, FileNotFoundError) as ex:
            self.warn(str(ex))
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

    def _toggle_state_changed(self, event):
        with contextlib.suppress(Exception):
            event['owner'].icon = 'check' if event['new'] else ''

    def _remove_stopwords_state_changed(self, *_):
        self._extra_stopwords.disabled = not self._remove_stopwords.value

    def setup(
        self, *, compute_callback: ComputeCallback, done_callback: DoneCallback, config: CorpusConfig = None
    ) -> "BaseGUI":
        self._config: CorpusConfig = config

        self._corpus_type.disabled = True
        self._filename_fields.disabled = True
        self._config_chooser.register_callback(self._config_chooser_changed)

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

        self.update_config()

        self.compute_callback = compute_callback
        self.done_callback = done_callback

        return self

    def _config_chooser_changed(self, *_):
        self._config = CorpusConfig.load(self.config_filename)
        if self._config is None:
            self.warn(f"Cannot load config {basename(self.config_filename)}")
        else:
            self.info(f"👌 Config {basename(self.config_filename)} loaded!")
            self.update_config()

    def update_config(self) -> "BaseGUI":
        if self._config is None:
            self.warn("No config loaded!")

            self._corpus_filename.filter_pattern = "*.*"
            self._corpus_filename.refresh()

            self._corpus_type.value = None
            self._filename_fields.value = ''

        else:
            self._corpus_filename.filter_pattern = self._config.corpus_pattern

            if self._config.corpus_source_exists() is not None:
                self._corpus_filename.reset(
                    path=dirname(self._config.pipeline_payload.source),
                    filename=basename(self._config.pipeline_payload.source),
                )

            self._corpus_filename.refresh()

            self._corpus_type.value = self._config.corpus_type
            self._filename_fields.value = ';'.join(self._config.text_reader_opts.filename_fields)
            self.update_pos_schema({})

        return self

    def update_pos_schema(self, *_) -> "BaseGUI":
        for _pos_widget in [self._pos_includes, self._pos_paddings, self._pos_excludes]:
            _pos_widget.value = []
            _pos_widget.options = {}

        if self._config is None:
            return self

        pos_schema: utility.PoS_Tag_Scheme = self._config.pos_schema

        tags: dict[str, str] = {f"{tag}/{description}": tag for tag, description in pos_schema.description.items()}

        for _pos_widget in [self._pos_includes, self._pos_paddings, self._pos_excludes]:
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
            transforms={
                'min-chars': 1,
                'only-alphabetic': self._only_alphabetic.value,
                'only-any-alphanumeric': self._only_any_alphanumeric.value,
                'to-lower': self._to_lowercase.value,
            }
        )

        if self._remove_stopwords.value:
            opts.remove_stopwords = self._config.language

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
            return join(self._target_folder.selected_path, self.corpus_tag)
        return self._target_folder.selected_path

    @property
    def config_folder(self) -> str:
        return self._config_chooser.selected_path

    @property
    def config_filename(self) -> str:
        return self._config_chooser.selected

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
    def phrases(self) -> list[list[str]]:
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
