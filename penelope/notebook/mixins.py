from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any, List, Mapping, Protocol, Set, Tuple, Union

from IPython.display import display as ipydisplay
from ipywidgets import Button, HBox, Layout, Output, SelectMultiple, ToggleButton
from penelope import utility as pu
from .widgets_utils import register_observer

from . import utility as nu

PivotKeySpec = Mapping[str, Union[str, Mapping[str, int]]]
PivotKeySpecArg = Union[List[PivotKeySpec], Mapping[str, List[PivotKeySpec]]]


class DownloadMixIn:
    def __init__(self) -> None:
        super().__init__()
        self._download = Button(description='Download', layout=Layout(width='auto'))
        self._download_output: Output = Output()
        self._download.on_click(self.download)

    def download(self, *_):
        with contextlib.suppress(Exception):
            with self._download_output:
                js_download = nu.create_js_download(self.data, index=True)
                if js_download is not None:
                    ipydisplay(js_download)


# pylint: disable=no-member
class IPivotKeysMixIn(Protocol):

    pivot_keys: pu.PivotKeys
    autoselect_key_values: bool

    def setup(self, **kwargs) -> "IPivotKeysMixIn":
        ...

    def reset(self) -> "IPivotKeysMixIn":
        ...

    @property
    def pivot_keys_text_names(self) -> List[str]:
        ...

    @property
    def pivot_keys_id_names(self) -> List[str]:
        ...

    @property
    def pivot_keys_filter_values(self) -> pu.PropertyValueMaskingOpts:
        ...

    @property
    def unstack_tabular(self) -> bool:
        ...

    def pivot_key_handler(self, change: dict, *_):
        ...

    def layout(self) -> HBox:
        ...

    def observe(self, value: bool) -> None:
        ...


class PivotKeysMixIn:
    """Defines controls and event logics for pivot keys and filters for pivot key values.
    super() must implement: __init__() setup() reset()  observe()
    Sibling class must implement: _display(), temporal_key: str
    """

    def __init__(self: IPivotKeysMixIn, pivot_key_specs: PivotKeySpecArg = None, **kwargs):
        super().__init__(**kwargs)

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

    def setup(self: IPivotKeysMixIn, **kwargs) -> "IPivotKeysMixIn":
        if self.pivot_keys.has_pivot_keys:
            self._unstack_tabular.observe(self._display, 'value')
        super().setup(**kwargs)
        return self

    def reset(self: IPivotKeysMixIn) -> "IPivotKeysMixIn":
        self.observe(False)
        self._pivot_keys_text_names.value = ['None']
        self._filter_keys.value = []
        self._unstack_tabular.value = False
        self.observe(True)
        super().reset()
        return self

    @property
    def pivot_keys_text_names(self: IPivotKeysMixIn) -> List[str]:
        """Return column names for selected the pivot keys"""
        return [x for x in self._pivot_keys_text_names.value if x != 'None']

    @property
    def pivot_keys_id_names(self: IPivotKeysMixIn) -> List[str]:
        """Return ID column names for selected pivot key"""
        return [self.pivot_keys.text_name2id_name.get(x) for x in self.pivot_keys_text_names]

    @property
    def pivot_keys_filter_values(self: IPivotKeysMixIn) -> pu.PropertyValueMaskingOpts:
        """Returns user's filter selections as a name-to-values mapping."""
        key_values = defaultdict(list)
        value_tuples: Tuple[str, str] = [x.split(': ') for x in self._filter_keys.value]
        for k, v in value_tuples:
            key_values[k].append(v)
        filter_opts = self.pivot_keys.create_filter_key_values_dict(key_values, decode=True)
        return filter_opts

    @property
    def unstack_tabular(self: IPivotKeysMixIn) -> bool:
        if len(self.pivot_keys_text_names) > 0:
            return self._unstack_tabular.value
        return False

    def pivot_key_handler(self: IPivotKeysMixIn, change: dict, *_):

        try:

            if self.prevent_event:
                return

            self.prevent_event = True

            # if 'None' in self._pivot_keys_text_names.value:
            #     self._pivot_keys_text_names.value = ['None']
            #     self._filter_keys.value = []
            #     self._filter_keys.options = []
            #     return

            old_keys: Set[str] = set(change['old']) - set(('None',))
            new_keys: Set[str] = set(change['new']) - set(('None',))

            # new_options: List[str] = set(self.pivot_keys.key_values_str(new_keys - old_keys, sep=': '))
            # remove_options: List[str] = set(self.pivot_keys.key_values_str(old_keys - new_keys, sep=': '))

            # self._filter_keys.value = []
            # self._filter_keys.options = sorted(list((set(self._filter_keys.options) - remove_options) | new_options))
            # self._filter_keys.value = sorted(list((set(self._filter_keys.value) - remove_options) | new_options))

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

    # def unstack_pivot_keys(self: IPivotKeysMixIn, data: pd.DataFrame) -> pd.DataFrame:
    #     """Unstacks a dataframe that has been grouped by PivotKeys"""
    #     if len(self.pivot_keys_text_names) > 0 and data is not None:
    #         data: pd.DataFrame = self.data.set_index([self.temporal_key] + self.pivot_keys_text_names)
    #         while isinstance(data.index, pd.MultiIndex):
    #             data = data.unstack(level=1, fill_value=0)
    #             if isinstance(data.columns, pd.MultiIndex):
    #                 data.columns = [' '.join(x) for x in data.columns]
    #     return data

    def observe(self: IPivotKeysMixIn, value: bool) -> None:
        register_observer(self._pivot_keys_text_names, handler=self.pivot_key_handler, value=value)
        display_trigger_ctrls: List[Any] = (
            [self._unstack_tabular, self._filter_keys]
            if self.pivot_keys.has_pivot_keys
            else [self._pivot_keys_text_names]
        )

        for ctrl in display_trigger_ctrls:
            register_observer(ctrl, handler=self._display, value=value)
        super().observe(value=value)
