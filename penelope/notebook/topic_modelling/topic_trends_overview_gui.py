from __future__ import annotations

from typing import Any, Callable

import ipywidgets as w
import pandas as pd
from IPython.display import display

from penelope import topic_modelling as tm
from penelope import utility as pu
from penelope.notebook import widgets_utils as wu

from .. import grid_utility as gu
from .. import widgets_utils
from . import mixins as mx
from . import model_container as mc
from .topic_trends_overview_gui_utility import display_heatmap

TEXT_ID = 'topic_relevance'


class TopicTrendsOverviewGUI(mx.AlertMixIn, mx.ComputeMixIn, mx.TopicsStateGui):
    def __init__(self, state: mc.TopicModelContainer):
        super().__init__(state=state)

        # FIXME, calculator: tm.MemoizedTopicPrevalenceOverTimeCalculator if caching....
        slider_opts = {
            'continuous_update': False,
            'layout': dict(width='140px'),
            'readout': False,
            'handle_color': 'lightblue',
        }
        timespan: tuple[int, int] = self.inferred_topics.timespan
        yearspan: tuple[int, int] = self.inferred_topics.startspan(10)

        self.titles: pd.DataFrame = None

        weighings = [(x['description'], x['key']) for x in tm.YEARLY_AVERAGE_COMPUTE_METHODS]

        self._text_id: str = TEXT_ID
        self._text: w.HTML = widgets_utils.text_widget(TEXT_ID)
        self._flip_axis: w.ToggleButton = w.ToggleButton(
            value=False, description='Flip', icon='', layout=dict(width="80px")
        )
        self._aggregate: w.Dropdown = w.Dropdown(options=weighings, value='max_weight', layout=dict(width="140px"))
        self._threshold_label: w.HTML = w.HTML("<b>Threshold</b>")
        self._threshold: w.FloatSlider = w.FloatSlider(min=0.01, max=1.0, value=0.05, step=0.01, **slider_opts)
        self._year_range_label: w.HTML = w.HTML("Years")
        self._year_range: w.IntRangeSlider = w.IntRangeSlider(
            min=timespan[0], max=timespan[1], step=1, value=yearspan, **slider_opts
        )

        self._output_format: w.Dropdown = w.Dropdown(
            options=['Heatmap', 'Table'], value='Heatmap', layout=dict(width="140px")
        )
        self._auto_compute.layout.width = "80px"
        self._output: w.Output = w.Output()
        self._content_placeholder: w.Box = None
        self._extra_placeholder: w.Box = None

    def setup(self, **kwargs) -> "TopicTrendsOverviewGUI":
        super().setup(**kwargs)
        self._compute_handler: Callable[[Any], None] = self.update_handler
        self.titles: pd.DataFrame = self.inferred_topics.get_topic_titles(n_tokens=100)
        self.observe_slider_update_label(self._year_range, self._year_range_label, "Years")
        self.observe_slider_update_label(self._threshold, self._threshold_label, "Threshold")
        self.observe(value=True, handler=self.update_handler)
        return self

    def observe(self, value: bool, **kwargs) -> TopicTrendsOverviewGUI:  # pylint: disable=arguments-differ
        super().observe(value=value, **kwargs)
        # value = value and self.auto_compute  # Never override autocompute
        wu.register_observer(self._aggregate, handler=self.update_handler, value=value)
        wu.register_observer(self._output_format, handler=self.update_handler, value=value)
        wu.register_observer(self._flip_axis, handler=self.update_handler, value=value)
        wu.register_observer(self._threshold, handler=self.update_handler, value=value)
        wu.register_observer(self._year_range, handler=self.update_handler, value=value)
        return self

    def layout(self) -> w.VBox:
        return w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                self._year_range_label,
                                self._year_range,
                                self._threshold_label,
                                self._threshold,
                            ]
                        ),
                    ]
                    + ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                    + [
                        w.VBox(
                            [
                                w.HBox([w.HTML("Aggregate"), self._aggregate]),
                                w.HTML("Output"),
                                w.HBox([self._output_format, self._flip_axis]),
                                w.HBox([self._compute, self._auto_compute]),
                                self._alert,
                            ]
                        ),
                    ]
                ),
                w.HBox([self._output]),
                w.HBox([self._text] + ([self._content_placeholder] if self._content_placeholder is not None else [])),
            ]
        )

    def update(self) -> pd.DataFrame:
        n_top_relevance: int = None
        yearly_weights: pd.DataFrame = (
            self.inferred_topics.calculator.reset()
            .threshold(self.threshold)
            .filter_by_keys(**self.filter_opts.opts)
            .yearly_topic_weights(self.get_result_threshold(), n_top_relevance=n_top_relevance)
            .value
        )
        return yearly_weights

    def update_handler(self, *_):

        self.alert("⌛ Computing...")
        self._output.clear_output()

        try:
            with self._output:

                try:
                    weights: pd.DataFrame = self.update()
                except pu.EmptyDataError:
                    weights = None

                if weights is None:
                    self.alert("😡 No data, please change filters..")
                elif self.output_format in ('xlsx', 'csv', 'clipboard'):
                    pu.ts_store(data=weights, extension=self.output_format, basename='heatmap_weights')
                elif self.output_format == "table":
                    g = gu.table_widget(weights)
                    display(g)
                else:
                    display_heatmap(
                        weights,
                        self.titles,
                        flip_axis=self._flip_axis.value,
                        aggregate=self.aggregate,
                        output_format=self.output_format,
                    )
            self.alert("✅")
        except Exception as ex:
            self.warn(f"😡 {ex}")

    @property
    def years(self) -> tuple[int, int]:
        return self._year_range.value

    @property
    def output_format(self) -> str:
        return self._output_format.value.lower()

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        return pu.PropertyValueMaskingOpts(year=self.years)

    @property
    def threshold(self) -> float:
        return self._threshold.value

    def get_result_threshold(self) -> float:
        return 0.0

    @property
    def aggregate(self) -> tuple:
        return self._aggregate.value


def display_gui(state: mc.TopicModelContainer) -> TopicTrendsOverviewGUI:
    gui: TopicTrendsOverviewGUI = TopicTrendsOverviewGUI(state=state).setup()
    display(gui.layout())
    gui.update_handler()
    return gui
