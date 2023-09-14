import contextlib
import os
from typing import Callable, List

import bokeh.plotting as bp
import ipywidgets as widgets
import pandas as pd
from loguru import logger

from penelope import pipeline
from penelope.corpus import DocumentIndex
from penelope.corpus.document_index import DocumentIndexHelper
from penelope.pipeline import checkpoint as cp
from penelope.pipeline import interfaces, tasks

from ...plot import plot_multiline, plot_stacked_bar
from ...utility import PoS_Tag_Scheme, path_add_suffix, strip_path_and_extension
from .. import grid_utility as gu
from ..utility import CLEAR_OUTPUT, OutputsTabExt

TOKEN_COUNT_GROUPINGS = ['decade', 'lustrum', 'year']

DEBUG_VIEW = widgets.Output()
# pylint: disable=too-many-instance-attributes


class TokenCountsGUI:
    """GUI component that displays word trends"""

    def __init__(
        self,
        compute_callback: Callable[["TokenCountsGUI", DocumentIndex], pd.DataFrame] = None,
    ):
        self.compute_callback: Callable[["TokenCountsGUI", DocumentIndex], pd.DataFrame] = (
            compute_callback or compute_token_count_data
        )

        self.document_index: DocumentIndex = None

        self._corpus_configs: widgets.Dropdown = widgets.Dropdown(
            description='', options=[], value=None, layout={'width': '200px'}
        )
        self._normalize: widgets.ToggleButton = widgets.ToggleButton(
            description="Normalize", icon='check', value=False, layout=widgets.Layout(width='140px')
        )
        self._smooth: widgets.ToggleButton = widgets.ToggleButton(
            description="Smooth", icon='check', value=False, layout=widgets.Layout(width='140px')
        )
        self._grouping: widgets.Dropdown = widgets.Dropdown(
            options=TOKEN_COUNT_GROUPINGS,
            value='year',
            description='',
            disabled=False,
            layout=widgets.Layout(width='90px'),
        )
        self._status: widgets.Label = widgets.Label(layout=widgets.Layout(width='50%', border="0px transparent white"))
        self._categories: widgets.SelectMultiple = widgets.SelectMultiple(
            options=[],
            value=[],
            rows=12,
            layout=widgets.Layout(width='120px'),
        )

        self._tab: OutputsTabExt = OutputsTabExt(["Table", "Line", "Bar"], layout={'width': '98%'})

    def layout(self) -> widgets.HBox:
        return widgets.HBox(
            [
                widgets.VBox([widgets.HTML("<b>PoS groups</b>"), self._categories], layout={'width': '140px'}),
                widgets.VBox(
                    [
                        widgets.HBox(
                            [self._normalize, self._smooth, self._grouping, self._corpus_configs, self._status]
                        ),
                        widgets.HBox([self._tab], layout={'width': '98%'}),
                        DEBUG_VIEW,
                    ],
                    layout={'width': '98%'},
                ),
            ],
            layout={'width': '98%'},
        )

    def _plot_counts(self, *_) -> None:
        try:
            if self.document_index is None:  # pragma: no cover
                self.alert("Please load a corpus!")
                return

            data = self.compute_callback(self, self.document_index)

            def plot_lines():
                p: bp.figure = plot_multiline(df=data.set_index(self.grouping), smooth=self.smooth)
                bp.show(p)

            def plot_bars():
                p: bp.figure = plot_stacked_bar(df=data.set_index(self.grouping))
                bp.show(p)

            self._tab.display_content(0, what=gu.table_widget(data), clear=True)
            self._tab.display_content(1, what=plot_lines, clear=True)
            self._tab.display_content(2, what=plot_bars, clear=True)

            self.alert("✔")

        except ValueError as ex:  # pragma: no cover
            self.alert(str(ex))
        except Exception as ex:  # pragma: no cover
            logger.exception(ex)
            self.warn(str(ex))

    def setup(self, config_filenames: List[str]) -> "TokenCountsGUI":
        self._corpus_configs.options = {strip_path_and_extension(path): path for path in config_filenames}
        self._corpus_configs.value = None

        if len(config_filenames) > 0:
            self._corpus_configs.value = config_filenames[0]

        self._categories.observe(self._plot_counts, names='value')
        self._normalize.observe(self._plot_counts, names='value')
        self._smooth.observe(self._plot_counts, names='value')
        self._grouping.observe(self._plot_counts, names='value')
        self._corpus_configs.observe(self._display_handler, names='value')

        return self

    def _display_handler(self, _):
        self.display()

    @property
    def config_filename(self) -> str:
        return self._corpus_configs.value

    @DEBUG_VIEW.capture(clear_output=CLEAR_OUTPUT)
    def display(self) -> "TokenCountsGUI":
        if self.config_filename is None:
            return self

        config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(self.config_filename)

        if not config.corpus_source_exists():
            self.alert(f"Config {config.corpus_name} has no specified corpus.")
            return self

        self.set_schema(config.pos_schema)

        try:
            self.document_index: DocumentIndex = load_document_index(config)

            if isinstance(self.document_index, pd.DataFrame):
                self._plot_counts()

        except Exception as ex:
            self.alert(f"failed: {ex}")

        return self

    def alert(self, msg: str):
        self._status.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")

    @property
    def smooth(self) -> bool:
        return self._smooth.value

    @property
    def normalize(self) -> bool:
        return self._normalize.value

    @property
    def grouping(self) -> str:
        return self._grouping.value

    @property
    def categories(self) -> List[str]:
        return list(self._categories.value)

    def set_schema(self, pos_schema: PoS_Tag_Scheme) -> None:
        self._categories.values = []
        self._categories.options = ['#Tokens'] + pos_schema.PD_PoS_groups.index.tolist()
        self._categories.values = ['#Tokens']


@DEBUG_VIEW.capture(clear_output=False)
def compute_token_count_data(args: TokenCountsGUI, document_index: DocumentIndex) -> pd.DataFrame:
    if len(args.categories or []) > 0:
        count_columns = list(args.categories)
    else:
        count_columns = [x for x in document_index.columns if x not in TOKEN_COUNT_GROUPINGS + ['#Tokens']]

    total = document_index.groupby(args.grouping)['#Tokens'].sum()
    data = document_index.groupby(args.grouping).sum()[count_columns]
    if args.normalize:
        data = data.div(total, axis=0)

    if args.smooth:
        data = data.interpolate(method='index')

    return data.reset_index()


def probe_checkpoint_document_index(pipe: pipeline.CorpusPipeline) -> pd.DataFrame:
    with contextlib.suppress(Exception):
        task: tasks.CheckpointFeather = pipe.find(tasks.CheckpointFeather)
        if task:
            return cp.feather.read_document_index(task.folder)

    with contextlib.suppress(Exception):
        task = pipe.find(tasks.LoadTaggedCSV)
        if task:
            return cp.feather.read_document_index(task.serialize_opts.feather_folder)

    return None


def load_by_pipeline(corpus_config: pipeline.CorpusConfig):
    """FIXME: This does not handle all cases"""
    if not corpus_config.pipeline_payload.source:
        logger.info("corpus filename is undefined. please check configuration")
        return None

    if not os.path.isfile(corpus_config.pipeline_payload.source):
        logger.info(f"corpus file {corpus_config.pipeline_payload.source} not found. please check configuration.")
        return None

    tagged_corpus_source: str = path_add_suffix(
        corpus_config.pipeline_payload.source, interfaces.DEFAULT_TAGGED_FRAMES_FILENAME_SUFFIX
    )

    p: pipeline.CorpusPipeline = corpus_config.get_pipeline(
        "tagged_frame_pipeline",
        tagged_corpus_source=tagged_corpus_source,
        enable_checkpoint=True,
    )

    document_index: DocumentIndex = probe_checkpoint_document_index(p)

    if document_index is None:
        p.exhaust()
        document_index: DocumentIndex = p.payload.document_index

    return document_index


@DEBUG_VIEW.capture(clear_output=CLEAR_OUTPUT)
def load_document_index(corpus_config: pipeline.CorpusConfig) -> pd.DataFrame:
    if corpus_config.pipeline_payload.document_index_source is not None:
        document_index: pd.DataFrame = DocumentIndexHelper.load(
            filename=corpus_config.pipeline_payload.document_index_source,
            sep=corpus_config.pipeline_payload.document_index_sep or ';',
        ).document_index

    else:
        document_index: pd.DataFrame = load_by_pipeline(corpus_config)

    if 'n_raw_tokens' not in document_index.columns:
        if 'n_tokens' in document_index.columns:
            document_index['n_raw_tokens'] = document_index['n_tokens']
        else:
            raise interfaces.PipelineError("expected required column `n_raw_tokens` not found")

    document_index['lustrum'] = document_index.year - document_index.year % 5
    document_index['decade'] = document_index.year - document_index.year % 10

    document_index = document_index.rename(columns={"n_raw_tokens": "#Tokens"}).fillna(0)

    pos_schema: PoS_Tag_Scheme = corpus_config.pos_schema

    groups = TOKEN_COUNT_GROUPINGS + ['#Tokens'] + pos_schema.PD_PoS_groups.keys().tolist()

    columns = [x for x in groups if x in document_index.columns]

    document_index = document_index[columns]

    return document_index
