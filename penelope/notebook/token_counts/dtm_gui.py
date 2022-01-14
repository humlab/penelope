from __future__ import annotations

import contextlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Mapping, Set, Tuple, Union

import pandas as pd
from ipydatagrid import DataGrid, TextRenderer
from IPython.display import display as ipydisplay
from ipywidgets import HTML, Button, Dropdown, HBox, Layout, Output, SelectMultiple, ToggleButton, VBox
from loguru import logger
from penelope import corpus as pc
from penelope import utility as pu

from ..utility import CLEAR_OUTPUT, FileChooserExt2, OutputsTabExt, create_js_download
from . import plot as pp

TEMPORAL_GROUP_BY = ['decade', 'lustrum', 'year']

DEBUG_VIEW = Output()

# pylint: disable=too-many-instance-attributes,too-many-public-methods


def observe(ctrl, handler, names: str = 'value', value: bool = True):
    with contextlib.suppress(Exception):
        getattr(ctrl, "observe" if value else "unobserve")(handler, names=names)


@dataclass
class ComputeOpts:
    source_folder: str
    document_index: pd.DataFrame
    normalize: bool
    smooth: bool
    pos_groups: List[str]
    temporal_key: str
    pivot_keys_id_names: List[str]
    pivot_keys_filter: pu.PropertyValueMaskingOpts
    unstack_tabular: bool


def plot_tabular(df: pd.DataFrame, opts: ComputeOpts) -> DataGrid:
    renderer = TextRenderer(format=',') if not opts.normalize else TextRenderer(format=',.4f')
    grid = DataGrid(
        df,
        base_row_size=30,
        selection_mode="cell",
        editable=False,
        base_column_size=80,
        # layout={'height': '200px'}
        renderers={c: renderer for c in df.columns if c not in TEMPORAL_GROUP_BY},
    )
    grid.transform([{"type": "sort", "columnIndex": 0, "desc": False}])
    grid.auto_fit_params = {"area": "all", "padding": 40, "numCols": 1}
    grid.auto_fit_columns = True
    return grid


@DEBUG_VIEW.capture(clear_output=False)
def compute(document_index: pd.DataFrame, opts: ComputeOpts) -> pd.DataFrame:

    di: pd.DataFrame = document_index

    if opts.pivot_keys_filter is not None:
        di = opts.pivot_keys_filter.apply(di)

    pivot_keys_id_names: List[str] = [opts.temporal_key] + list(opts.pivot_keys_id_names)

    count_columns: List[str] = (
        list(opts.pos_groups)
        if len(opts.pos_groups or []) > 0
        else [x for x in di.columns if x not in TEMPORAL_GROUP_BY + ['Total'] + pivot_keys_id_names]
    )
    data: pd.DataFrame = di.groupby(pivot_keys_id_names).sum()[count_columns]

    if opts.normalize:
        total: pd.Series = di.groupby(pivot_keys_id_names)['Total'].sum()
        data = data.div(total, axis=0)

    if opts.smooth:
        data = data.interpolate(method='index')

    data = data.reset_index()[pivot_keys_id_names + count_columns]
    return data


@DEBUG_VIEW.capture(clear_output=CLEAR_OUTPUT)
def prepare_document_index(document_index: str, keep_columns: List[str]) -> pd.DataFrame:
    """Prepares document index by adding/renaming columns

    Args:
        source (str): document index source
        columns (List[str]): PoS-groups column names
    """

    if 'n_raw_tokens' not in document_index.columns:
        raise ValueError("expected required column `n_raw_tokens` not found")

    document_index['lustrum'] = document_index.year - document_index.year % 5
    document_index['decade'] = document_index.year - document_index.year % 10
    document_index = document_index.rename(columns={"n_raw_tokens": "Total"}).fillna(0)

    """strip away irrelevant columns"""
    groups = TEMPORAL_GROUP_BY + ['Total'] + sorted(keep_columns)
    keep_columns = [x for x in groups if x in document_index.columns]
    document_index = document_index[keep_columns]

    return document_index


PivotKeySpec = Mapping[str, Union[str, Mapping[str, int]]]


class BasicDTMGUI:
    """GUI component that displays token counts"""

    def __init__(
        self,
        *,
        default_folder: str,
        pivot_key_specs: List[PivotKeySpec] | Mapping[str, List[PivotKeySpec]] = None,
        **defaults: dict,
    ):
        """GUI base for PoS token count statistics."""

        self.pivot_keys: pu.PivotKeys = pu.PivotKeys(pivot_key_specs)
        self.default_folder: str = default_folder
        self.document_index: pd.DataFrame = None
        self.data: pd.DataFrame = None
        self.defaults: dict = defaults
        self.PoS_tag_groups: pd.DataFrame = pu.PD_PoS_tag_groups
        self._source_folder: FileChooserExt2 = None

        self._normalize: ToggleButton = ToggleButton(
            description="Normalize", icon='check', value=defaults.get('normalize', False), layout=Layout(width='140px')
        )
        self._smooth: ToggleButton = ToggleButton(
            description="Smooth", icon='check', value=defaults.get('smooth', False), layout=Layout(width='140px')
        )
        self._unstack_tabular: ToggleButton = ToggleButton(
            description="Unstack", icon='check', value=defaults.get('unstack', False), layout=Layout(width='140px')
        )
        self._temporal_key: Dropdown = Dropdown(
            options=TEMPORAL_GROUP_BY,
            value=defaults.get('temporal_key', 'decade'),
            description='',
            disabled=False,
            layout=Layout(width='90px'),
        )
        self._status: HTML = HTML(layout=Layout(width='50%', border="0px transparent white"))
        self._pos_groups: SelectMultiple = SelectMultiple(
            options=['Total'] + [x for x in self.PoS_tag_groups.index.tolist() if x != "Delimiter"],
            value=['Total'],
            rows=10,
            layout=Layout(width='120px'),
        )

        self._pivot_keys_text_names: SelectMultiple = SelectMultiple(
            options=['None'] + list(self.pivot_keys.text_names),
            value=['None'],
            rows=5,
            layout=Layout(width='120px'),
        )
        self._filter_keys: SelectMultiple = SelectMultiple(options=[], value=[], rows=12, layout=Layout(width='120px'))

        self._tab: OutputsTabExt = OutputsTabExt(["Table", "Line", "Bar"], layout={'width': '98%'})
        self._widgets_placeholder: HBox = HBox()
        self._download = Button(description='Download', layout=Layout(width='auto'))
        self._download_output: Output = Output()

    def setup(self, load_data: bool = False) -> "BasicDTMGUI":
        self.observe(False)
        self._source_folder: FileChooserExt2 = FileChooserExt2(
            path=self.default_folder,
            title='<b>Corpus folder</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=True,
        )
        self._source_folder.refresh()
        self._source_folder.register_callback(self._load)
        self._download.on_click(self.download)

        if len(self.pivot_keys.text_names) > 0:
            self._unstack_tabular.observe(self._display, 'value')

        self.observe(True)
        if load_data:
            self._load()
        return self

    def reset(self) -> "BasicDTMGUI":
        self.observe(False)
        self.document_index = None
        self.data = None
        self._normalize.value = self.defaults.get('normalize', False)
        self._smooth.value = self.defaults.get('smooth', False)
        self._temporal_key.value = self.defaults.get('temporal_key', 'decade')
        self._pos_groups.value = ['Total']
        self._pivot_keys_text_names.value = ['None']
        self._filter_keys.value = []
        self._unstack_tabular.value = False
        self.observe(True)

    def download(self, *_):
        with contextlib.suppress(Exception):
            with self._download_output:
                js_download = create_js_download(self.data, index=True)
                if js_download is not None:
                    ipydisplay(js_download)

    def plot_tabular(self, df: pd.DataFrame, opts: ComputeOpts) -> DataGrid:
        return plot_tabular(df, opts)

    def compute(self) -> pd.DataFrame:
        self.data = compute(self.document_index, self.opts)
        return self.data

    @property
    def normalize(self) -> bool:
        return self._normalize.value

    @property
    def smooth(self) -> bool:
        return self._smooth.value

    @property
    def temporal_key(self) -> str:
        return self._temporal_key.value

    @property
    def pivot_keys_text_names(self) -> List[str]:
        """Return column names for user-selected pivot keys"""
        return [x for x in self._pivot_keys_text_names.value if x != 'None']

    @property
    def pivot_keys_id_names(self) -> List[str]:
        """Return ID column names for user selected pivot key"""
        return [self.pivot_keys.text_name2id_name.get(x) for x in self.pivot_keys_text_names]

    @property
    def pivot_keys_filter_values(self) -> pu.PropertyValueMaskingOpts:
        """Returns user's filter selections as a name-to-values mapping."""
        key_values = defaultdict(list)
        value_tuples: Tuple[str, str] = [x.split(': ') for x in self._filter_keys.value]
        for k, v in value_tuples:
            key_values[k].append(v)
        filter_opts = self.pivot_keys.create_filter_key_values_dict(key_values, decode=True)
        return filter_opts

    @property
    def selected_pos_groups(self) -> List[str]:
        return self._pos_groups.value

    @property
    def source_folder(self) -> str:
        return self._source_folder.selected_path

    @property
    def unstack_tabular(self) -> bool:
        if len(self.pivot_keys_text_names) > 0:
            return self._unstack_tabular.value
        return False

    @property
    def opts(self) -> ComputeOpts:
        return ComputeOpts(
            source_folder=self.source_folder,
            document_index=self.document_index,
            normalize=self.normalize,
            smooth=self.smooth,
            pos_groups=self.selected_pos_groups,
            temporal_key=self.temporal_key,
            pivot_keys_id_names=self.pivot_keys_id_names,
            pivot_keys_filter=self.pivot_keys_filter_values,
            unstack_tabular=self.unstack_tabular,
        )

    def layout(self) -> HBox:
        return VBox(
            [
                VBox([self._source_folder]),
                HBox(
                    [
                        VBox(
                            [
                                HTML("<b>PoS groups</b>"),
                                self._pos_groups,
                            ]
                            + (
                                []
                                if len(self.pivot_keys.text_names) == 0
                                else [
                                    HTML("<b>Pivot by</b>"),
                                    self._pivot_keys_text_names,
                                ]
                            )
                            + (
                                []
                                if len(self.pivot_keys.text_names) == 0
                                else [
                                    HTML("<b>Filter by</b>"),
                                    self._filter_keys,
                                ]
                            ),
                            layout={'width': '140px'},
                        ),
                        VBox(
                            [
                                HBox(
                                    [
                                        self._normalize,
                                        self._smooth,
                                    ]
                                    + ([self._unstack_tabular] if len(self.pivot_keys.text_names) > 0 else [])
                                    + [
                                        self._temporal_key,
                                        self._widgets_placeholder,
                                        self._download,
                                        self._status,
                                        self._download_output,
                                    ]
                                ),
                                HBox(
                                    [
                                        self._tab,
                                    ],
                                    layout={'width': '98%'},
                                ),
                                DEBUG_VIEW,
                            ],
                            layout={'width': '98%'},
                        ),
                    ],
                    layout={'width': '98%'},
                ),
            ]
        )

    def pivot_key_handler(self, change: dict, *_):

        old_keys: Set[str] = set(change['old']) - set(('None',))
        new_keys: Set[str] = set(change['new']) - set(('None',))

        # if 'None' in self._pivot_keys_text_names.value:
        #     self._pivot_keys_text_names.value = ['None']
        #     self._filter_keys.value = []
        #     self._filter_keys.options = []
        #     return

        new_options: List[str] = set(self.pivot_keys.key_values_str(new_keys - old_keys, sep=': '))
        remove_options: List[str] = set(self.pivot_keys.key_values_str(old_keys - new_keys, sep=': '))

        self._filter_keys.values = []
        self._filter_keys.options = sorted(list((set(self._filter_keys.options) - remove_options) | new_options))
        self._filter_keys.values = sorted(list((set(self._filter_keys.value) - remove_options) | new_options))

    def observe(self, value: bool) -> None:

        observe(self._pivot_keys_text_names, handler=self.pivot_key_handler, value=value)

        display_trigger_ctrls: List[Any] = [self._pos_groups, self._normalize, self._smooth, self._temporal_key] + (
            [self._unstack_tabular, self._filter_keys]
            if len(self.pivot_keys.text_names) > 0
            else [self._pivot_keys_text_names]
        )

        for ctrl in display_trigger_ctrls:
            observe(ctrl, handler=self._display, value=value)

    def _display(self, _):
        self.observe(False)
        try:
            self.display()
        finally:
            self.observe(True)

    @DEBUG_VIEW.capture(clear_output=False)
    def display(self) -> "BasicDTMGUI":
        try:
            data: pd.DataFrame = self.compute()
            if self.document_index is None:
                self.alert("Please select a corpus folder!")
            else:
                self.plot(data)
        except Exception as ex:
            self.alert(f"failed: {ex}")
        return self

    @DEBUG_VIEW.capture(clear_output=False)
    def load(self, source: Union[str, pd.DataFrame]) -> "BasicDTMGUI":
        self.document_index = (
            source if isinstance(source, pd.DataFrame) else pc.VectorizedCorpus.load_document_index(source)
        )
        return self

    def keep_columns(self) -> List[str]:
        return self.PoS_tag_groups.index.tolist()

    @DEBUG_VIEW.capture(clear_output=False)
    def prepare(self) -> "BasicDTMGUI":
        self.document_index = prepare_document_index(self.document_index, keep_columns=self.keep_columns())
        return self

    def _load(self, *_) -> None:
        try:
            self.load(source=self.opts.source_folder)
            self.prepare()
            self.display()
        except FileNotFoundError as ex:
            self.alert(ex)
        except ValueError as ex:
            self.alert(ex)

    def unstack_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if len(self.pivot_keys_text_names) > 0 and self.data is not None:
            data: pd.DataFrame = self.data.set_index([self.temporal_key] + self.pivot_keys_text_names)
            while isinstance(data.index, pd.MultiIndex):
                data = data.unstack(level=1, fill_value=0)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [' '.join(x) for x in data.columns]
        return data

    @DEBUG_VIEW.capture(clear_output=False)
    def plot(self, data: pd.DataFrame) -> None:

        try:

            data: pd.DataFrame = self.compute()

            if len(self.pivot_keys_text_names) > 0:
                unstacked_data: pd.DataFrame = self.unstack_data(data)
            else:
                unstacked_data = data

            table: pd.DataFrame = self.plot_tabular(unstacked_data if self.unstack_tabular else data, self.opts)

            if self.temporal_key not in unstacked_data.columns:
                unstacked_data = unstacked_data.reset_index()  # set_index(self.temporal_key)

            plot_data: pd.DataFrame = unstacked_data.set_index(self.temporal_key)  # , drop=False).rename_axis('')

            # FIXME: Fix Smooth!!
            plot_lines = lambda: pp.plot_multiline(df=plot_data, smooth=False)
            plot_bar = lambda: pp.plot_stacked_bar(df=plot_data)

            # FIXME: Add option to plot several graphs?

            self._tab.display_content(0, what=table, clear=True)
            self._tab.display_content(1, what=plot_lines, clear=True)
            self._tab.display_content(2, what=plot_bar, clear=True)

            self.alert("✔")
        except ValueError as ex:
            self.alert(str(ex))
        except Exception as ex:
            logger.exception(ex)
            self.warn(str(ex))

    def alert(self, msg: str):
        self._status.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")
