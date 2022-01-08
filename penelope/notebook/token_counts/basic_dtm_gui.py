import contextlib
from dataclasses import dataclass
from typing import List, Union

import pandas as pd
from ipydatagrid import DataGrid, TextRenderer
from ipywidgets import HTML, Dropdown, HBox, Label, Layout, Output, SelectMultiple, ToggleButton, VBox
from loguru import logger
from penelope import corpus as pc
from penelope import utility as pu

from ..utility import CLEAR_OUTPUT, FileChooserExt2, OutputsTabExt
from .plot import plot_by_bokeh as plot_dataframe

TEMPORAL_GROUP_BY = ['decade', 'lustrum', 'year']

DEBUG_VIEW = Output()

# pylint: disable=too-many-instance-attributes


@dataclass
class ComputeOpts:
    source_folder: str
    document_index: pd.DataFrame
    normalize: bool
    smooth: bool
    pos_groups: List[str]
    temporal_key: str
    grouping_keys: List[str]


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
def compute(document_index: pd.DataFrame, args: ComputeOpts) -> pd.DataFrame:

    grouping_columns: List[str] = [args.temporal_key] + list(args.grouping_keys)
    count_columns: List[str] = (
        list(args.pos_groups)
        if len(args.pos_groups or []) > 0
        else [x for x in document_index.columns if x not in TEMPORAL_GROUP_BY + ['Total']]
    )

    data: pd.DataFrame = document_index.groupby(grouping_columns).sum()[count_columns]

    if args.normalize:
        total: pd.Series = document_index.groupby(grouping_columns)['Total'].sum()
        data = data.div(total, axis=0)

    if args.smooth:
        data = data.interpolate(method='index')

    return data.reset_index()


@DEBUG_VIEW.capture(clear_output=CLEAR_OUTPUT)
def prepare_document_index(document_index: str, columns: List[str]) -> pd.DataFrame:
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
    groups = TEMPORAL_GROUP_BY + ['Total'] + sorted(columns)
    columns = [x for x in groups if x in document_index.columns]
    document_index = document_index[columns]

    return document_index


class BasicDTMGUI:
    """GUI component that displays token counts"""

    def plot_tabular(self, df: pd.DataFrame, opts: ComputeOpts) -> DataGrid:
        return plot_tabular(df, opts)

    def compute(self, df: pd.DataFrame, opts: ComputeOpts) -> pd.DataFrame:
        return compute(df, opts)

    def __init__(self, default_folder: str, avaliable_grouping_keys: List[str] = None):
        """GUI base for PoS token count statistics."""

        self.avaliable_grouping_keys: List[str] = avaliable_grouping_keys or []
        self.document_index: pd.DataFrame = None
        self.default_folder: str = default_folder
        self.PoS_tag_groups: pd.DataFrame = pu.PD_PoS_tag_groups
        self._source_folder: FileChooserExt2 = None

        self._normalize: ToggleButton = ToggleButton(
            description="Normalize", icon='check', value=False, layout=Layout(width='140px')
        )
        self._smooth: ToggleButton = ToggleButton(
            description="Smooth", icon='check', value=False, layout=Layout(width='140px')
        )
        self._temporal_key: Dropdown = Dropdown(
            options=TEMPORAL_GROUP_BY,
            value='year',
            description='',
            disabled=False,
            layout=Layout(width='90px'),
        )
        self._status: Label = Label(layout=Layout(width='50%', border="0px transparent white"))
        self._pos_groups: SelectMultiple = SelectMultiple(
            options=['Total'] + self.PoS_tag_groups.index.tolist(),
            value=[],
            rows=12,
            layout=Layout(width='120px'),
        )

        self._grouping_keys: SelectMultiple = SelectMultiple(
            options=self.avaliable_grouping_keys,
            value=[],
            rows=12,
            layout=Layout(width='120px'),
        )
        self.tab: OutputsTabExt = OutputsTabExt(["Table", "Plot"], layout={'width': '98%'})
        self._widgets_placeholder: HBox = HBox()

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
    def grouping_keys(self) -> List[str]:
        return self._grouping_keys.value

    @property
    def selected_pos_groups(self) -> List[str]:
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
            grouping_keys=self.grouping_keys,
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
                                if len(self.avaliable_grouping_keys) == 0
                                else [
                                    HTML("<b>Pivot by</b>"),
                                    self._grouping_keys,
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
                                        self._temporal_key,
                                        self._widgets_placeholder,
                                        self._status,
                                    ]
                                ),
                                HBox(
                                    [
                                        self.tab,
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

    def setup(self) -> "BasicDTMGUI":
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
        self._load()

        return self

    def observe(self, value: bool) -> None:
        method: str = 'observe' if value else 'unobserve'
        for ctrl in [self._pos_groups, self._normalize, self._smooth, self._temporal_key]:
            with contextlib.suppress(Exception):
                getattr(ctrl, method)(self._display, names='value')

    def _display(self, _):
        self.observe(False)
        try:
            self.display()
        finally:
            self.observe(True)

    @DEBUG_VIEW.capture(clear_output=False)
    def display(self) -> "BasicDTMGUI":
        try:
            self.plot()
        except Exception as ex:
            self.alert(f"failed: {ex}")
        return self

    @DEBUG_VIEW.capture(clear_output=False)
    def load(self, source: Union[str, pd.DataFrame]) -> None:
        self.document_index = (
            source if isinstance(source, pd.DataFrame) else pc.VectorizedCorpus.load_document_index(source)
        )

    @DEBUG_VIEW.capture(clear_output=False)
    def prepare(self) -> None:
        self.document_index = prepare_document_index(self.document_index, columns=self.PoS_tag_groups.index.tolist())

    def _load(self, *_) -> None:
        try:
            self.load(source=self.opts.source_folder)
            self.prepare()
            self.display()
        except FileNotFoundError as ex:
            self.alert(ex)
        except ValueError as ex:
            self.alert(ex)

    @DEBUG_VIEW.capture(clear_output=False)
    def plot(self) -> None:

        try:
            if self.document_index is None:  # pragma: no cover
                self.alert("Please select a corpus folder!")
                return

            data: pd.DataFrame = self.compute(self.document_index, self.opts)

            plot_table = self.plot_tabular(data, self.opts)
            plot_graph = lambda: plot_dataframe(data_source=data.set_index(self.temporal_key), smooth=self.smooth)

            self.tab.display_content(0, what=plot_table, clear=True)
            self.tab.display_content(1, what=plot_graph, clear=True)

            self.alert("✔")

        except ValueError as ex:  # pragma: no cover
            self.alert(str(ex))
        except Exception as ex:  # pragma: no cover
            logger.exception(ex)
            self.warn(str(ex))

    def alert(self, msg: str):
        self._status.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")
