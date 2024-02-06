from __future__ import annotations

import contextlib
import typing as t
from collections import defaultdict

import ipywidgets as w
import pandas as pd
from IPython.display import display as ipydisplay

from penelope import utility as pu
from penelope.corpus.render import IRenderService, ITextRepository
from penelope.pipeline.config import CorpusConfig, DependencyError
from penelope.plot.colors import get_color_palette

from . import utility as nu
from .widgets_utils import register_observer

PivotKeySpec = dict[str, str | dict[str, int]]
PivotKeySpecArg = list[PivotKeySpec] | dict[str, list[PivotKeySpec]] | str

# pylint: disable=no-member


class DownloadMixIn:
    def __init__(self) -> None:
        super().__init__()
        self._download: w.Button = w.Button(description='Download', layout=dict(width='auto'))
        self._download_output: w.Output = w.Output()
        self._download.on_click(self.download)
        self._download_data_name: str = 'data'

    def download(self, *_):
        with contextlib.suppress(Exception):
            with self._download_output:
                data = getattr(self, self._download_data_name)
                js_download = nu.create_js_download(data, index=True)
                if js_download is not None:
                    ipydisplay(js_download)


class PivotKeysMixIn:
    """Defines controls and event logics for pivot keys and filters for pivot key values."""

    def __init__(self, pivot_key_specs: PivotKeySpecArg = None, **kwargs):
        self.prevent_event: bool = False

        super().__init__(**kwargs)

        self._display_event_handler: t.Callable[[t.Any], None] = None

        self._single_pivot_key_picker: w.Dropdown = w.Dropdown(options={}, value=None, layout=dict(width='100px'))
        self._filter_keys_picker: w.SelectMultiple = w.SelectMultiple(
            options=[], value=[], rows=6, layout=dict(width='100px')
        )
        self._filter_values_picker: w.SelectMultiple = w.SelectMultiple(
            options=[], value=[], rows=12, layout=dict(width='120px')
        )

        self._unstack_tabular: w.ToggleButton = w.ToggleButton(
            description="Unstack", icon='check', value=False, layout=dict(width='140px')
        )

        self.clear_label: str = '(clear)'
        self.pivot_key_specs: PivotKeySpecArg = pivot_key_specs
        self.pivot_keys = pivot_key_specs

    @property
    def pivot_keys(self) -> pu.PivotKeys:
        return self._pivot_keys

    def clamp(self, value: int, min_value: int, max_value: int) -> int:
        return max(min(value, max_value), min_value)

    @pivot_keys.setter
    def pivot_keys(self, value: pu.PivotKeys | dict):
        if isinstance(value, str):
            if value.endswith('.yaml') or value.endswith('.yml'):
                value: dict = pu.PivotKeys.load(value)

        self._pivot_keys = value if isinstance(value, pu.PivotKeys) else pu.PivotKeys(value)

        if value is None:
            return

        _prevent_event: bool = self.prevent_event
        self.prevent_event = True

        self._single_pivot_key_picker.value = None
        self._single_pivot_key_picker.options = self.pivot_keys.single_key_options
        self._single_pivot_key_picker.value = next(iter(self._single_pivot_key_picker.options.values()), None)

        self._filter_keys_picker.value = []
        self._filter_keys_picker.options = [self.clear_label] + list(self.pivot_keys.text_names)
        self._filter_keys_picker.value = [self.clear_label]
        self._filter_keys_picker.rows = self.clamp(len(self._filter_keys_picker.options), 2, 6)

        self._filter_values_picker.value = []
        self._filter_values_picker.options = []

        self._filter_values_picker.rows = self.clamp(len(self._filter_values_picker.options), 4, 12)

        self.autoselect_key_values: bool = False
        self.prevent_event = _prevent_event

    # @property
    # def _filter_key_options(self) -> list[str]:
    #     """Avaliable filter key values"""
    #     return self.filter_keys.options

    @property
    def selected_filter_values(self) -> list[str]:
        """Selected filter key values"""
        return self._filter_values_picker.value

    def setup(self, **kwargs) -> "PivotKeysMixIn":
        if hasattr(super(), 'setup'):
            getattr(super(), 'setup')(**kwargs)
        register_observer(self._filter_keys_picker, handler=self.pivot_key_handler, value=True)
        return self

    def reset(self) -> "PivotKeysMixIn":
        self.observe(value=False, handler=self._display_event_handler)
        self._filter_keys_picker.value = [self.clear_label]
        self._filter_values_picker.value = []
        self._unstack_tabular.value = False
        self.observe(value=True, handler=self._display_event_handler)

        if hasattr(super(), 'reset'):
            super().reset()

        return self

    # region Single pivot key picker (dropdown)

    @property
    def picked_pivot_id(self) -> str:
        """Returns ID of picked pivot key (single picked dropdown)"""
        return self._single_pivot_key_picker.value

    @property
    def picked_pivot_name(self) -> str:
        """Returns NAME of picked pivot key (single picked dropdown)"""
        return self.pivot_keys.key_id2key_name[self.picked_pivot_id]

    @property
    def picked_pivot_value_mapping(self) -> dict[int, str]:
        """Returns ID to VALUE-NAME mapping of picked pivot key (single picked dropdown)"""
        return self._pivot_keys.key_value_id2name(self.picked_pivot_name)

    # endregion

    @property
    def pivot_keys_text_names(self) -> list[str]:
        """Return column names for selected the pivot keys"""
        return [x for x in self._filter_keys_picker.value if x not in (self.clear_label, '')]

    @property
    def pivot_keys_id_names(self) -> list[str]:
        """Return ID column names for selected pivot key"""
        return [self.pivot_keys.key_name2key_id.get(x) for x in self.pivot_keys_text_names]

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        """Returns user's filter selections as a name-to-values mapping."""
        key_values = defaultdict(list)
        value_tuples: tuple[str, str] = [x.split(': ') for x in self._filter_values_picker.value]
        for k, v in value_tuples:
            key_values[k].append(v)
        filter_opts = self.pivot_keys.create_filter_key_values_dict(key_values, decode=True)
        if hasattr(super(), 'filter_opts'):
            filter_opts.update(super().filter_opts)
        return filter_opts

    @property
    def unstack_tabular(self) -> bool:
        if len(self.pivot_keys_text_names) > 0:
            return self._unstack_tabular.value
        return False

    def decode_pivot_keys(self, df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        return self.pivot_keys.decode_pivot_keys(df, drop)

    def pivot_key_handler(self, change: dict, *_):
        """Pivot key selection has changed"""
        try:
            if self.prevent_event:
                return

            self.prevent_event = True

            old_keys: set[str] = set(change['old']) - set((self.clear_label,))
            new_keys: set[str] = set(change['new']) - set((self.clear_label,))

            add_options: set[str] = set(self._pivot_keys.key_values_str(new_keys - old_keys, sep=': '))
            del_options: set[str] = set(self._pivot_keys.key_values_str(old_keys - new_keys, sep=': '))

            ctrl_options: set[str] = (set(self._filter_values_picker.options) - del_options) | add_options
            current_values: set[str] = set(self._filter_values_picker.value)
            ctrl_values: set[str] = (current_values - del_options) | (
                add_options if self.autoselect_key_values else set()
            )

            values_changed: bool = ctrl_values != current_values

            if values_changed:
                self._filter_values_picker.value = []

            self._filter_values_picker.options = sorted(list(ctrl_options))

            if values_changed:
                self._filter_values_picker.value = sorted(list(ctrl_values))

        finally:
            self.prevent_event = False

    def display_trigger_ctrls(self) -> list[w.Widget]:
        return (
            [self._unstack_tabular, self._filter_values_picker]
            if self.pivot_keys.has_pivot_keys
            else [self._filter_keys_picker]
        )

    def observe(self, value: bool, *, handler: t.Callable[[t.Any], None], **kwargs) -> None:
        if handler is None:
            return

        def wrapper_handler(change: dict, *_):
            prevent_event = self.prevent_event
            if self.prevent_event:
                return
            try:
                handler(change, *_)
            finally:
                self.prevent_event = prevent_event

        self._display_event_handler = wrapper_handler

        for ctrl in self.display_trigger_ctrls():
            register_observer(ctrl, handler=wrapper_handler, value=value)

        if hasattr(super(), "observe"):
            getattr(super(), "observe")(value=value, handler=wrapper_handler, **kwargs)

    def default_pivot_keys_layout(self, vertical: bool = False, **kwargs) -> w.Widget:
        # if not self._pivot_keys.has_pivot_keys:
        #     return w.VBox()
        width: str = '200px'
        if kwargs.get('rows') is not None:
            self._filter_values_picker.rows = kwargs.get('rows')
        self._filter_values_picker.layout.width = '95%'
        self._filter_keys_picker.layout.width = '95%'
        if vertical:
            return w.VBox(
                [
                    w.HTML("<b>Filter by</b>"),
                    self._filter_keys_picker,
                    self._filter_values_picker,
                ],
                layout=dict(width=width),
            )
        return w.HBox(
            [
                w.VBox([w.HTML("<b>Filter by</b>"), self._filter_keys_picker], layout={'width': '95%'}),
                w.VBox([w.HTML("<b>Value</b>"), self._filter_values_picker], layout={'width': '95%'}),
            ],
            layout=dict(width=width),
        )


class MultiLinePivotKeysMixIn(PivotKeysMixIn):
    def __init__(self, pivot_key_specs: t.Any = None, color_presets: dict[str, str] = None, **kwargs):
        super().__init__(pivot_key_specs, **kwargs)
        self._line_name: w.Text = w.Text(description="", placeholder="(name)", layout=dict(width='80px'))
        self._add_line: w.Button = w.Button(description="\u2714")
        self._del_line: w.Button = w.Button(description="\u2718")
        self._lines: w.Dropdown = w.Dropdown()

        self._color_presets = color_presets
        self._color_palette: str = get_color_palette()
        self._next_color: w.Button = w.Button(description="\u2B6E")
        self._line_color: w.ColorPicker = w.ColorPicker(concise=True, description="", value=next(self._color_palette))
        self._line_color_preset_picker: w.Dropdown = w.Dropdown(description="", value=None)

    def default_pivot_keys_layout(self, vertical: bool = False, **kwargs) -> w.Widget:
        width: str = kwargs.get('width', '120px')

        self._filter_values_picker.rows = kwargs.get('rows', 12)
        self._filter_values_picker.layout = kwargs.get("layout", dict(width=width))
        self._line_color.layout = dict(width='32px')
        self._next_color.layout = dict(width='32px')

        if isinstance(self._color_presets, dict):
            self._line_color_preset_picker.layout = dict(width='45px')
            self._line_color_preset_picker.options = list(self._color_presets.items())
            self._line_color_preset_picker.value = None

        # self._single_pivot_key_picker.layout = kwargs.get('layout', dict(width=width))
        return w.VBox(
            [
                w.HBox([self._lines, self._del_line]),
                self._filter_values_picker,
                w.HBox(
                    [
                        w.HTML("Color "),
                        self._line_color,
                        self._next_color,
                        self._line_color_preset_picker,
                    ]
                ),
                w.HBox([w.HTML("Legend "), self._line_name, self._add_line]),
            ]
        )

    def display_trigger_ctrls(self) -> list[w.Widget]:
        return []

    def setup(self, **kwargs) -> "PivotKeysMixIn":
        if hasattr(super(), 'setup'):
            getattr(super(), 'setup')(**kwargs)
        self._add_line.on_click(self._add_line_callback)
        self._del_line.on_click(self._del_line_callback)
        self._next_color.on_click(self._next_color_callback)
        self._line_color_preset_picker.observe(self._set_color_by_preset)
        self._lines.observe(self._line_selected_callback, type='change')
        return self

    def _set_color_by_preset(self, *_):
        if self._line_color_preset_picker.value:
            self._line_color.value = self._line_color_preset_picker.value
            self._line_color_preset_picker.value = None

    def _next_color_callback(self, *_):
        self._line_color.value = next(self._color_palette)

    def _add_line_callback(self, *_):
        self.add_line(name=self.line_name, color=self.line_color, values=self.selected_filter_values)
        self._show_line(name="", color=next(self._color_palette), values=[])

    def add_line(self, name: str, color: str, values: list[str]):
        if not name:
            self.alert("😡 you must give the line a name")
            return
        if not values:
            self.alert("😡 please select value(s) that define the line")
            return
        self._lines.options = list(self._lines.options or []) + [(name, (name, color, values))]
        self.alert(f"✅ {name} added!")

    def _del_line_callback(self, *_):
        if not self._lines.options:
            self.alert("😡 no lines to delete")
            return

        if not self._lines.value:
            self.alert("😡 please select line to delete")

        options = list(self._lines.options)
        name = options[self._lines.index][0]
        del options[self._lines.index]
        self._lines.value = None
        self._lines.options = options
        self.alert(f"✅ {name} deleted")

        if options:
            self._lines.index = 0

    def _line_selected_callback(self, *_):
        if self._lines.value:
            self._show_line(*self._lines.options[self._lines.index][1])

    def _show_line(self, name: str, color: str, values: list[str]):
        self._line_name.value = name
        self._line_color.value = color
        self._filter_values_picker.value = values

    @property
    def line_name(self) -> str:
        return self._line_name.value

    @property
    def line_color(self) -> str:
        return self._line_color.value

    @property
    def lines(self) -> list[tuple[str, str, str]]:
        return [x[1] for x in self._lines.options or []]


class TextRepositoryMixIn:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._text_output: w.HTML = w.HTML(layout={'width': '50%', 'background-color': 'lightgreen'})
        self._content_placeholder: w.VBox = self._text_output

        self._content_type: t.Literal['raw', 'text', 'html'] = 'html'

        self._document_click_handler = self.on_row_click

    @property
    def text_output(self) -> w.HTML:
        return self._text_output

    def _resolve_key(self, key: str) -> t.Any:
        config: CorpusConfig = getattr(self, 'config', None)
        if config:
            try:
                return config.resolve_dependency(key)
            except DependencyError as ex:
                self.alert(str(ex))
                raise ex
        return None

    @property
    def render_service(self) -> IRenderService:
        return self._resolve_key('render_text')

    @property
    def text_repository(self) -> ITextRepository:
        return self._resolve_key('text_repository')

    @property
    def content_type(self) -> t.Literal['raw', 'text', 'html']:
        return self._content_type

    @content_type.setter
    def content_type(self, value: t.Literal['raw', 'text', 'html']):
        self._content_type = value

    def on_row_click(self, item: pd.Series, g: t.Any):  # pylint: disable=unused-argument
        try:
            document_name: str = item.get('document_name')
            data: dict = self.text_repository.get(document_name)
            self._text_output.value = self.render_service.render(data, kind=self.content_type)
        except Exception as ex:
            self._text_output.value = str(ex)
