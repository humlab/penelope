from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any, Callable, List, Mapping, Set, Tuple, Union

import ipywidgets as w
import pandas as pd
from IPython.display import display as ipydisplay

from penelope import utility as pu

from . import utility as nu
from .widgets_utils import register_observer

PivotKeySpec = Mapping[str, Union[str, Mapping[str, int]]]
PivotKeySpecArg = Union[List[PivotKeySpec], Mapping[str, List[PivotKeySpec]]]


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

        super().__init__(**kwargs)

        self._display_event_handler: Callable[[Any], None] = None

        self.pivot_keys: pu.PivotKeys = (
            pivot_key_specs if isinstance(pivot_key_specs, pu.PivotKeys) else pu.PivotKeys(pivot_key_specs)
        )

        """Single-select"""
        single_key_options: dict = {v['text_name']: v['id_name'] for v in self.pivot_keys.pivot_keys.values()}
        self._single_pivot_key_picker: w.Dropdown = w.Dropdown(
            options=single_key_options,
            value=next(iter(single_key_options.values())) if single_key_options else None,
            layout=dict(width='100px'),
        )

        """Multi-select"""
        self._multi_pivot_keys_picker: w.SelectMultiple = w.SelectMultiple(
            options=['None'] + list(self.pivot_keys.text_names),
            value=['None'],
            rows=5,
            layout=dict(width='100px'),
        )
        self._filter_keys: w.SelectMultiple = w.SelectMultiple(
            options=[], value=[], rows=12, layout=dict(width='120px')
        )

        self._unstack_tabular: w.ToggleButton = w.ToggleButton(
            description="Unstack", icon='check', value=False, layout=dict(width='140px')
        )
        self.autoselect_key_values: bool = False
        self.prevent_event: bool = False

    @property
    def filter_key_values(self) -> List[str]:
        """Avaliable filter key values"""
        return self._filter_keys.options

    @property
    def filter_key_selected_values(self) -> List[str]:
        """Avaliable filter key values"""
        return self._filter_keys.value

    def setup(self, **kwargs) -> "PivotKeysMixIn":
        if hasattr(super(), 'setup'):
            getattr(super(), 'setup')(**kwargs)
        register_observer(self._multi_pivot_keys_picker, handler=self.pivot_key_handler, value=True)
        return self

    def reset(self) -> "PivotKeysMixIn":

        self.observe(value=False, handler=self._display_event_handler)
        self._multi_pivot_keys_picker.value = ['None']
        self._filter_keys.value = []
        self._unstack_tabular.value = False
        self.observe(value=False, handler=self._display_event_handler)

        if hasattr(super(), 'reset'):
            super().reset()

        return self

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
        return self.pivot_keys.key_value_id2name(self.picked_pivot_name)

    @property
    def pivot_keys_text_names(self) -> List[str]:
        """Return column names for selected the pivot keys"""
        return [x for x in self._multi_pivot_keys_picker.value if x != 'None']

    @property
    def pivot_keys_id_names(self) -> List[str]:
        """Return ID column names for selected pivot key"""
        return [self.pivot_keys.key_name2key_id.get(x) for x in self.pivot_keys_text_names]

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        """Returns user's filter selections as a name-to-values mapping."""
        key_values = defaultdict(list)
        value_tuples: Tuple[str, str] = [x.split(': ') for x in self._filter_keys.value]
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

            old_keys: Set[str] = set(change['old']) - set(('None',))
            new_keys: Set[str] = set(change['new']) - set(('None',))

            add_options: Set[str] = set(self.pivot_keys.key_values_str(new_keys - old_keys, sep=': '))
            del_options: Set[str] = set(self.pivot_keys.key_values_str(old_keys - new_keys, sep=': '))

            ctrl_options: Set[str] = (set(self._filter_keys.options) - del_options) | add_options
            current_values: Set[str] = set(self._filter_keys.value)
            ctrl_values: Set[str] = (current_values - del_options) | (
                add_options if self.autoselect_key_values else set()
            )

            values_changed: bool = ctrl_values != current_values

            if values_changed:
                self._filter_keys.value = []

            self._filter_keys.options = sorted(list(ctrl_options))

            if values_changed:
                self._filter_keys.value = sorted(list(ctrl_values))

        finally:
            self.prevent_event = False

    def observe(self, value: bool, *, handler: Callable[[Any], None], **kwargs) -> None:

        if handler is None:
            return

        self._display_event_handler = handler

        display_trigger_ctrls: List[Any] = (
            [self._unstack_tabular, self._filter_keys, self._unstack_tabular]
            if self.pivot_keys.has_pivot_keys
            else [self._multi_pivot_keys_picker]
        )

        for ctrl in display_trigger_ctrls:
            register_observer(ctrl, handler=handler, value=value)

        if hasattr(super(), "observe"):
            getattr(super(), "observe")(value=value, handler=handler, **kwargs)

    def default_pivot_keys_layout(self, vertical: bool = False, **kwargs) -> w.Widget:
        width: str = kwargs.get('width', '100px')
        self._filter_keys.rows = kwargs.get('rows', 12)
        self._filter_keys.layout = kwargs.get('layout', dict(width='120px'))
        self._multi_pivot_keys_picker.layout = kwargs.get('layout', dict(width=width))
        if vertical:
            return w.VBox(
                [
                    w.HTML("<b>Filter by</b>"),
                    self._multi_pivot_keys_picker,
                    # w.HTML("<b>Value</b>"),
                    self._filter_keys,
                ]
            )
        return w.HBox(
            [
                w.VBox([w.HTML("<b>Filter by</b>"), self._multi_pivot_keys_picker]),
                w.VBox([w.HTML("<b>Value</b>"), self._filter_keys]),
            ]
        )
