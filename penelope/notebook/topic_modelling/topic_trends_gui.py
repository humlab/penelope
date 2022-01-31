from __future__ import annotations

from typing import Any, Callable

import ipywidgets as w  # type: ignore
import pandas as pd
from IPython.display import display

import penelope.topic_modelling as tm
import penelope.utility as pu
from penelope.notebook import widgets_utils as wu
from .utility import table_widget

from . import mixins as mx
from . import topic_trends_gui_utility as gui_utils
from .model_container import TopicModelContainer


class TopicTrendsGUI(mx.NextPrevTopicMixIn, mx.AlertMixIn, mx.ComputeMixIn, mx.TopicsStateGui):
    def __init__(self, state: TopicModelContainer):
        super().__init__(state=state)

        self.yearly_topic_weights: pd.DataFrame = None

        timespan: tuple[int, int] = self.inferred_topics.timespan

        self._text: w.HTML = w.HTML()

        weighings = [(x['description'], x['key']) for x in tm.YEARLY_AVERAGE_COMPUTE_METHODS]

        self._aggregate: w.Dropdown = w.Dropdown(
            description='Aggregate', options=weighings, value='true_average_weight', layout=dict(width="200px")
        )

        self._year_range: w.IntRangeSlider = w.IntRangeSlider(
            min=timespan[0], max=timespan[1], continues_update=False, value=timespan
        )
        self._threshold: w.FloatSlider = w.FloatSlider(min=0.01, max=1.0, value=0.05, step=0.01)

        self._output_format: w.Dropdown = w.Dropdown(
            description='Format',
            options=['Chart', 'Table', 'xlsx', 'csv', 'clipboard', 'pandas'],
            value='Chart',
            layout=dict(width="200px"),
        )
        self._output: w.Output = w.Output()
        self._compute_handler: Callable[[Any], None] = self.update_handler
        self._content_placeholder: w.Box = None
        self._extra_placeholder: w.VBox = w.HBox()

    def setup(self, **kwargs) -> "TopicTrendsGUI":
        super().setup(**kwargs)

        self.topic_id = (0, self.inferred_n_topics - 1)
        self.observe(value=True, handler=self.update_handler)
        return self

    def observe(self, value: bool, **kwargs) -> TopicTrendsGUI:  # pylint: disable=unused-argument
        wu.register_observer(self._topic_id, handler=self.topic_changed, value=value)
        wu.register_observer(self._aggregate, handler=self.display_handler, value=value)
        wu.register_observer(self._output_format, handler=self.display_handler, value=value)
        return self

    def layout(self):
        return w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                w.HBox([self._next_prev_layout]),
                                w.HTML("<b>Year range</b>"),
                                self._year_range,
                                w.HTML("<b>Threshold</b>"),
                                self._threshold,
                            ]
                        ),
                    ]
                    + ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                    + [
                        w.VBox(
                            [
                                self._aggregate,
                                w.HBox([self._output_format]),
                                self._alert,
                                w.HBox([self._compute, self._auto_compute]),
                            ]
                        ),
                    ]
                ),
                self._output,
                w.HBox([self._text] + ([self._content_placeholder] if self._content_placeholder is not None else [])),
            ]
        )

    def topic_changed(self, *_):
        if len(self.yearly_topic_weights) == 0:
            return
        self._text.value = f'ID {self.topic_id}: {self.inferred_topics.get_topic_title(self.topic_id, n_tokens=200)}'
        gui_utils.display_topic_trends(
            weight_over_time=self.yearly_topic_weights[(self.yearly_topic_weights.topic_id == self.topic_id)],
            year_range=self.years,
            value_column=self.aggregate,
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
        try:
            self.yearly_topic_weights = self.update()
            self.alert("✅")
        except pu.EmptyDataError:
            self.yearly_topic_weights = None
        except Exception as ex:
            self.warn(f"😡 {ex}")

        self.display_handler()

    def display_handler(self, *_):
        self._output.clear_output()
        try:
            with self._output:
                if self.yearly_topic_weights is None:
                    self.alert("😡 No data, please change filters..")
                elif self.output_format in ('xlsx', 'csv', 'clipboard'):
                    pu.ts_store(
                        data=self.yearly_topic_weights, extension=self.output_format, basename='heatmap_weights'
                    )
                elif self.output_format == "pandas":
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        display(self.yearly_topic_weights)
                elif self.output_format == "table":
                    g = table_widget(self.yearly_topic_weights, handler=self.click_handler)
                    display(g)
                else:
                    self.topic_changed()
            self.alert("✅")
        except Exception as ex:
            self.warn(f"😡 {ex}")

    def click_handler(self, item: pd.Series, _: Any) -> None:
        self.alert(f"You clicked:  {item['year']} {item['topic_id']}")

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


def display_gui(state: TopicModelContainer) -> TopicTrendsGUI:
    gui = TopicTrendsGUI(state=state).setup()
    display(gui.layout())
    gui.update_handler()
    return gui
