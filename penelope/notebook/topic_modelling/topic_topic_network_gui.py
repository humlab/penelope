from __future__ import annotations

from typing import Any, Callable

import ipywidgets as w  # type: ignore
import pandas as pd
from IPython.display import display

import penelope.utility as pu
from penelope.notebook import widgets_utils as wu

from . import mixins as mx
from .model_container import TopicModelContainer
from .topic_topic_network_gui_utility import display_topic_topic_network
from .utility import table_widget

# bokeh.plotting.output_notebook()
TEXT_ID = 'nx_topic_topic'
LAYOUT_OPTIONS = ['Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
OUTPUT_OPTIONS = {'Network': 'network', 'Table': 'table', 'Excel': 'XLSX', 'CSV': 'CSV', 'Clipboard': 'clipboard'}

# pylint: disable=too-many-instance-attributes


class TopicTopicGUI(mx.AlertMixIn, mx.ComputeMixIn, mx.TopicsStateGui):
    def __init__(self, state: TopicModelContainer):

        super().__init__(state=state)

        slider_opts = {
            'continuous_update': False,
            'layout': dict(width='140px'),
            'readout': False,
            'handle_color': 'lightblue',
        }

        self.network_data: pd.DataFrame = None
        self.topic_proportions: pd.DataFrame = None
        self.titles: pd.DataFrame = None

        timespan: tuple[int, int] = self.inferred_topics.timespan
        yearspan: tuple[int, int] = self.inferred_topics.startspan(10)

        self._text: w.HTML = wu.text_widget(TEXT_ID)
        self._year_range_label: w.HTML = w.HTML("Years")
        self._year_range: w.IntRangeSlider = w.IntRangeSlider(
            min=timespan[0], max=timespan[1], step=1, value=yearspan, **slider_opts
        )
        self._scale_label: w.HTML = w.HTML("<b>Scale</b>")
        self._scale: w.FloatSlider = w.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.1, **slider_opts)
        self._threshold_label: w.HTML = w.HTML("<b>Threshold</b>")
        self._threshold: w.FloatSlider = w.FloatSlider(min=0.01, max=1.0, value=0.20, step=0.01, **slider_opts)
        self._n_docs_label: w.HTML = w.HTML("<b>Documents in common</b>")
        self._n_docs: w.IntSlider = w.IntSlider(min=1, max=100, step=1, value=10, **slider_opts)
        self._network_layout: w.Dropdown = w.Dropdown(
            options=LAYOUT_OPTIONS, value='Fruchterman-Reingold', layout=dict(width='140px')
        )
        _ignore_options = [('', None)] + [('Topic #' + str(i), i) for i in range(0, self.inferred_n_topics)]
        self._ignores: w.SelectMultiple = w.SelectMultiple(
            options=_ignore_options, value=[], rows=10, layout=dict(width='100px')
        )
        self._node_range_label: w.HTML = w.HTML("<b>Node size</b>")
        self._node_range: w.IntRangeSlider = w.IntRangeSlider(min=10, max=100, step=1, value=(20, 60), **slider_opts)
        self._edge_range_label: w.HTML = w.HTML("<b>Edge size</b>")
        self._edge_range: w.IntRangeSlider = w.IntRangeSlider(min=1, max=20, step=1, value=(2, 6), **slider_opts)
        self._output_format: w.Dropdown = w.Dropdown(
            description='', options=OUTPUT_OPTIONS, value='network', layout=dict(width='140px')
        )
        self._output: w.Output = w.Output()
        self._content_placeholder: w.Box = None
        self._extra_placeholder: w.Box = w.VBox()

    def setup(self, **kwargs) -> "TopicTopicGUI":
        super().setup(**kwargs)
        self._compute_handler: Callable[[Any], None] = self.update_handler
        self.topic_proportions: pd.DataFrame = self.inferred_topics.calculator.reset().topic_proportions()
        self.titles: pd.DataFrame = self.inferred_topics.get_topic_titles()
        self.observe_slider_update_label(self._year_range, self._year_range_label, "Years")
        self.observe_slider_update_label(self._threshold, self._threshold_label, "Threshold")
        self.observe_slider_update_label(self._n_docs, self._n_docs_label, "Common docs")
        self.observe_slider_update_label(self._node_range, self._node_range_label, "Node size")
        self.observe_slider_update_label(self._edge_range, self._edge_range_label, "Edge size")
        self.observe_slider_update_label(self._scale, self._scale_label, "Scale")
        self.observe(value=True, handler=self.update_handler)
        return self

    def observe(self, value: bool, **kwargs) -> TopicTopicGUI:  # pylint: disable=unused-argument,arguments-differ
        wu.register_observer(self._threshold, handler=self.update_handler, value=value)
        wu.register_observer(self._n_docs, handler=self.update_handler, value=value)
        wu.register_observer(self._year_range, handler=self.update_handler, value=value)
        wu.register_observer(self._scale, handler=self.display_handler, value=value)
        wu.register_observer(self._node_range, handler=self.display_handler, value=value)
        wu.register_observer(self._edge_range, handler=self.display_handler, value=value)
        wu.register_observer(self._output_format, handler=self.display_handler, value=value)
        wu.register_observer(self._network_layout, handler=self.display_handler, value=value)
        wu.register_observer(self._ignores, handler=self.update_handler, value=value)
        return self

    def layout(self) -> w.VBox:
        extra_widgets: w.VBox = self.extra_widgets()

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
                                self._n_docs_label,
                                self._n_docs,
                            ]
                        ),
                        w.VBox(
                            [
                                w.HTML("<b>Ignore topics</b>"),
                                self._ignores,
                            ]
                        ),
                    ]
                    + ([extra_widgets] if extra_widgets else [])
                    + [
                        w.VBox(
                            [
                                self._node_range_label,
                                self._node_range,
                                self._edge_range_label,
                                self._edge_range,
                                self._scale_label,
                                self._scale,
                            ]
                        ),
                        w.VBox(
                            [
                                w.HTML("<b>Network layout</b>"),
                                self._network_layout,
                                w.HTML("<b>Output</b>"),
                                self._output_format,
                                self._auto_compute,
                                self._compute,
                                self._alert,
                            ]
                        ),
                    ]
                ),
                self._output,
                w.HBox([self._text] + ([self._content_placeholder] if self._content_placeholder is not None else [])),
            ]
        )

    def extra_widgets(self) -> w.VBox:
        return self._extra_placeholder

    def update(self) -> pd.DataFrame:

        topic_topic: pd.DataFrame = (
            self.inferred_topics.calculator.reset()
            .filter_by_keys(**self.filter_opts.opts)
            .threshold(threshold=self.threshold)
            .filter_by_topics(topic_ids=self.ignores, negate=True)
            .to_topic_topic_network(self.n_docs)
            .value
        )

        if len(topic_topic) == 0:
            raise pu.EmptyDataError()

        return topic_topic

    def update_handler(self, *_):

        self.alert("⌛ Computing...")
        try:
            self.network_data = self.update()
            self.alert("✅")
        except pu.EmptyDataError:
            self.network_data = None
        except Exception as ex:
            self.warn(f"😡 {ex}")

        self.display_handler()

    def display_handler(self, *_):

        self._output.clear_output()
        with self._output:

            if self.network_data is None:
                self.alert("😡 No data, please change filters..")
            elif self.output_format in ('xlsx', 'csv', 'clipboard'):
                pu.ts_store(data=self.network_data, extension=self.output_format, basename='heatmap_weights')
            elif self.output_format == "table":
                g = table_widget(self.network_data)
                display(g)
            else:

                display_topic_topic_network(
                    data=self.network_data,
                    layout=self._network_layout.value,
                    scale=self._scale.value,
                    node_range=self._node_range.value,
                    edge_range=self._edge_range.value,
                    element_id=TEXT_ID,
                    titles=self.titles,
                    topic_proportions=self.topic_proportions,
                )

    def click_handler(self, item: pd.Series, _: Any) -> None:
        self.alert(f"You clicked:  {item['year']} {item['topic_id']}")

    @property
    def years(self) -> tuple[int, int]:
        return self._year_range.value

    @property
    def output_format(self) -> str:
        return self._output_format.value.lower()

    @property
    def ignores(self) -> list[int]:
        return self._ignores.value

    @property
    def n_docs(self) -> int:
        return self._n_docs.value

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        return pu.PropertyValueMaskingOpts(year=self.years)

    @property
    def threshold(self) -> float:
        return self._threshold.value


def display_gui(state: TopicModelContainer) -> TopicModelContainer:
    gui: TopicTopicGUI = TopicTopicGUI(state=state).setup()
    display(gui.layout())
    gui.update_handler()
    return gui
