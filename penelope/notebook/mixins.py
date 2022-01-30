from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any, Callable, List, Mapping, Set, Tuple, Union

import pandas as pd
from IPython.display import display as ipydisplay
from ipywidgets import Button, Layout, Output, SelectMultiple, ToggleButton

from penelope import utility as pu

from . import utility as nu
from .widgets_utils import register_observer

PivotKeySpec = Mapping[str, Union[str, Mapping[str, int]]]
PivotKeySpecArg = Union[List[PivotKeySpec], Mapping[str, List[PivotKeySpec]]]


class DownloadMixIn:
    def __init__(self) -> None:
        super().__init__()
        self._download = Button(description='Download', layout=Layout(width='auto'))
        self._download_output: Output = Output()
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

        self._pivot_keys_text_names: SelectMultiple = SelectMultiple(
            options=['None'] + list(self.pivot_keys.text_names),
            value=['None'],
            rows=5,
            layout=Layout(width='120px'),
        )
        self._filter_keys: SelectMultiple = SelectMultiple(options=[], value=[], rows=12, layout=Layout(width='120px'))

        self._unstack_tabular: ToggleButton = ToggleButton(
            description="Unstack", icon='check', value=False, layout=Layout(width='140px')
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
        register_observer(self._pivot_keys_text_names, handler=self.pivot_key_handler, value=True)
        return self

    def reset(self) -> "PivotKeysMixIn":

        self.observe(value=False, handler=self._display_event_handler)
        self._pivot_keys_text_names.value = ['None']
        self._filter_keys.value = []
        self._unstack_tabular.value = False
        self.observe(value=False, handler=self._display_event_handler)

        if hasattr(super(), 'reset'):
            super().reset()

        return self

    @property
    def pivot_keys_text_names(self) -> List[str]:
        """Return column names for selected the pivot keys"""
        return [x for x in self._pivot_keys_text_names.value if x != 'None']

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
            else [self._pivot_keys_text_names]
        )

        for ctrl in display_trigger_ctrls:
            register_observer(ctrl, handler=handler, value=value)

        if hasattr(super(), "observe"):
            getattr(super(), "observe")(value=value, handler=handler, **kwargs)
