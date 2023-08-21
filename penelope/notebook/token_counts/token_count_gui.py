from __future__ import annotations

from typing import Any, Callable, Self

import pandas as pd
from ipydatagrid import DataGrid, TextRenderer
from ipywidgets import HTML, Dropdown, HBox, Layout, Output, SelectMultiple, ToggleButton, VBox
from loguru import logger

from penelope import corpus as pc
from penelope import utility as pu
from penelope.plot import plot_multiline, plot_stacked_bar

from ..mixins import DownloadMixIn, PivotKeysMixIn, PivotKeySpec
from ..utility import FileChooserExt2, OutputsTabExt
from ..widgets_utils import register_observer
from .compute import TEMPORAL_GROUP_BY, ComputeOpts, compute_statistics, prepare_document_index

DEBUG_VIEW = Output()

# pylint: disable=too-many-instance-attributes,too-many-public-methods


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


class BaseTokenCountGUI(DownloadMixIn):
    """GUI component that displays token counts"""

    def __init__(self, *, default_folder: str, **defaults):
        """GUI base for PoS token count statistics."""
        super().__init__()
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

        self._tab: OutputsTabExt = OutputsTabExt(["Table", "Line", "Bar"], layout={'width': '98%'})
        self._widgets_placeholder: HBox = HBox(children=[])
        self._sidebar_placeholder: HBox = HBox(children=[])

    def setup(self, **kwargs) -> Self:
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
        self.observe(True)
        if kwargs.get('load_data', False):
            self._load()
        return self

    def reset(self) -> Self:
        self.observe(False)
        self.document_index = None
        self.data = None
        self._normalize.value = self.defaults.get('normalize', False)
        self._smooth.value = self.defaults.get('smooth', False)
        self._temporal_key.value = self.defaults.get('temporal_key', 'decade')
        self._pos_groups.value = ['Total']
        self.observe(True)
        return self

    def plot_tabular(self, df: pd.DataFrame, opts: ComputeOpts) -> DataGrid:
        return plot_tabular(df, opts)

    def compute(self) -> pd.DataFrame:
        self.data = compute_statistics(self.document_index, self.opts)
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
    def selected_pos_groups(self) -> list[str]:
        return self._pos_groups.value

    @property
    def source_folder(self) -> str:
        return self._source_folder.selected_path

    @property
    def opts(self) -> ComputeOpts:
        return ComputeOpts(
            source_folder=self.source_folder,
            document_index=self.document_index,
            normalize=self.normalize,
            smooth=self.smooth,
            pos_groups=self.selected_pos_groups,
            temporal_key=self.temporal_key,
        )

    def layout(self) -> HBox:
        return VBox(
            [
                VBox([self._source_folder]),
                HBox(
                    [
                        VBox(
                            [HTML("<b>PoS groups</b>"), self._pos_groups]
                            + list(self._sidebar_placeholder.children or []),
                            layout={'width': '140px'},
                        ),
                        VBox(
                            [
                                HBox(
                                    [
                                        self._temporal_key,
                                        self._normalize,
                                        self._smooth,
                                        self._widgets_placeholder,
                                        self._download,
                                        self._status,
                                        self._download_output,
                                    ]
                                ),
                                HBox([self._tab], layout={'width': '98%'}),
                                DEBUG_VIEW,
                            ],
                            layout={'width': '98%'},
                        ),
                    ],
                    layout={'width': '98%'},
                ),
            ]
        )

    def observe(
        self, value: bool, handler: Callable[[Any], Any] = None, **kwargs  # pylint: disable=unused-argument
    ) -> None:
        display_trigger_ctrls: list[Any] = [self._pos_groups, self._normalize, self._smooth, self._temporal_key]
        for ctrl in display_trigger_ctrls:
            register_observer(ctrl, handler=handler or self._display_handler, value=value)

    def _display_handler(self, _):
        self.observe(False)
        try:
            self.display()
        finally:
            self.observe(True)

    @DEBUG_VIEW.capture(clear_output=False)
    def display(self) -> Self:
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
    def load(self, source: str | pd.DataFrame) -> Self:
        self.document_index = (
            source if isinstance(source, pd.DataFrame) else pc.VectorizedCorpus.load_document_index(source)
        )
        return self

    def keep_columns(self) -> list[str]:
        return self.PoS_tag_groups.index.tolist()

    @DEBUG_VIEW.capture(clear_output=False)
    def prepare(self) -> Self:
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

    def prepare_plot_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        table_data: pd.DataFrame = self.plot_tabular(data, self.opts)
        plot_data: pd.DataFrame = data

        return (table_data, plot_data)

    @DEBUG_VIEW.capture(clear_output=False)
    def plot(self, data: pd.DataFrame) -> None:
        try:
            table_data, plot_data = self.prepare_plot_data(data=data)

            if self.temporal_key not in plot_data.columns:
                plot_data = plot_data.reset_index()

            plot_data: pd.DataFrame = plot_data.set_index(self.temporal_key)

            fx_lines = lambda: plot_multiline(df=plot_data, smooth=False)  # FIXME: Fix Smooth!!
            fx_bar = lambda: plot_stacked_bar(df=plot_data)

            self._tab.display_content(0, what=table_data, clear=True)
            self._tab.display_content(1, what=fx_lines, clear=True)
            self._tab.display_content(2, what=fx_bar, clear=True)

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


# pylint: disable= useless-super-delegation
class TokenCountGUI(PivotKeysMixIn, BaseTokenCountGUI):
    def __init__(self, pivot_key_specs: list[PivotKeySpec] | dict[str, list[PivotKeySpec]] = None, **kwargs):
        super().__init__(pivot_key_specs, **kwargs)

    def unstack_pivot_keys(self, data: pd.DataFrame) -> pd.DataFrame:
        return pu.unstack_data(data, [self.temporal_key] + self.pivot_keys_text_names)

    def prepare_plot_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        unstacked_data: pd.DataFrame = self.unstack_pivot_keys(data) if len(self.pivot_keys_text_names) > 0 else data

        table_data: pd.DataFrame = self.plot_tabular(unstacked_data if self.unstack_tabular else data, self.opts)
        plot_data: pd.DataFrame = unstacked_data

        return (table_data, plot_data)

    @property
    def opts(self) -> ComputeOpts:
        opts: ComputeOpts = super().opts
        opts.pivot_keys_id_names = self.pivot_keys_id_names
        opts.filter_opts = self.filter_opts
        opts.unstack_tabular = self.unstack_tabular
        return opts

    def layout(self) -> HBox:
        self._sidebar_placeholder.children = (
            list(self._sidebar_placeholder.children or [])
            + ([] if not self.pivot_keys.has_pivot_keys else [HTML("<b>Pivot by</b>"), self._filter_keys_picker])
            + ([] if not self.pivot_keys.has_pivot_keys else [HTML("<b>Filter by</b>"), self._filter_values_picker])
        )
        self._widgets_placeholder.children = list(self._widgets_placeholder.children or []) + (
            [self._unstack_tabular] if len(self.pivot_keys.text_names) > 0 else []
        )

        return super().layout()

    def setup(self, **kwargs) -> Self:
        return super().setup(**kwargs)

    def load(self, source: str | pd.DataFrame) -> Self:
        super().load(source)

        if isinstance(source, str):
            self.pivot_keys.load_by_probe(source)

        return self

    def observe(
        self, value: bool, handler: Callable[[Any], None] = None, **kwargs
    ) -> None:  # pylint: disable=arguments-differ
        super().observe(value=value, handler=handler or self._display_handler, **kwargs)
