from __future__ import annotations

import uuid
from typing import Any, Callable

import ipywidgets as w
import pandas as pd
from IPython.display import display

import penelope.utility as pu
from penelope.network.bipartite_plot import plot_bipartite_dataframe
from penelope.notebook import mixins as ox
from penelope.notebook import topic_modelling as ntm
from penelope.notebook import widgets_utils as wu
from penelope.notebook.topic_modelling import mixins as mx

LAYOUT_OPTIONS = ['Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
OUTPUT_OPTIONS = {'Network': 'network', 'Table': 'table', 'Excel': 'XLSX', 'CSV': 'CSV', 'Clipboard': 'clipboard'}


# pylint: disable=too-many-locals, too-many-arguments, too-many-instance-attributes


class PivotTopicNetworkGUI(ox.PivotKeysMixIn, mx.AlertMixIn, mx.ComputeMixIn, ntm.TopicsStateGui):
    def __init__(self, pivot_key_specs: ox.PivotKeySpecArg, state: ntm.TopicModelContainer, **kwargs):
        super().__init__(pivot_key_specs, state=state, **kwargs)

        slider_opts = {
            'continuous_update': False,
            'layout': dict(width='140px'),
            'readout': False,
            'handle_color': 'lightblue',
        }

        self.text_id: str = f"{str(uuid.uuid4())[:6]}"
        self.network_data: pd.DataFrame = None
        self.topic_proportions: pd.DataFrame = None
        self.titles: pd.DataFrame = self.inferred_topics.get_topic_titles()

        timespan: tuple[int, int] = self.inferred_topics.timespan
        yearspan: tuple[int, int] = self.inferred_topics.startspan(10)

        self._text: w.HTML = wu.text_widget(self.text_id)
        self._year_range_label: w.HTML = w.HTML("Years")
        self._year_range: w.IntRangeSlider = w.IntRangeSlider(
            min=timespan[0], max=timespan[1], step=1, value=yearspan, **slider_opts
        )
        self._scale_label: w.HTML = w.HTML("Scale")
        self._scale: w.FloatSlider = w.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.1, **slider_opts)
        self._threshold_label: w.HTML = w.HTML("<b>Threshold</b>")
        self._threshold = w.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.05, **slider_opts)
        self._grouped_threshold_label: w.HTML = w.HTML("<b>Threshold (p)</b>")
        self._grouped_threshold = w.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.10, **slider_opts)
        self._network_layout: w.Dropdown = w.Dropdown(
            options=LAYOUT_OPTIONS, value='Fruchterman-Reingold', layout=dict(width='140px')
        )
        _ignore_options = [('', None)] + [('Topic #' + str(i), i) for i in range(0, self.inferred_n_topics)]
        self._ignores: w.SelectMultiple = w.SelectMultiple(
            options=_ignore_options, value=[], rows=13, layout=dict(width='120px'), style={'background': '#c9c9c9'}
        )
        self._aggregate: w.Dropdown = w.Dropdown(
            description='Aggregate', options=['mean', 'max'], value='mean', layout=dict(width="200px")
        )
        self._output_format: w.Dropdown = w.Dropdown(
            description='', options=OUTPUT_OPTIONS, value='network', layout=dict(width='140px')
        )
        self._output: w.Output = w.Output()
        self._content_placeholder: w.Box = None
        self._extra_placeholder: w.Box = w.VBox()

    def setup(self, **kwargs) -> "PivotTopicNetworkGUI":
        super().setup(**kwargs)
        self._compute_handler: Callable[[Any], None] = self.update_handler
        self.topic_proportions: pd.DataFrame = self.inferred_topics.calculator.reset().topic_proportions()
        self.titles: pd.DataFrame = self.inferred_topics.get_topic_titles()
        self.observe_slider_update_label(self._year_range, self._year_range_label, "Years")
        self.observe_slider_update_label(self._threshold, self._threshold_label, "Threshold")
        self.observe_slider_update_label(self._grouped_threshold, self._grouped_threshold_label, "Threshold (g)")
        self.observe_slider_update_label(self._scale, self._scale_label, "Scale")
        self.observe(value=True, handler=self.update_handler)
        self.inferred_topics.calculator.reset()
        return self

    def observe(self, value: bool, **kwargs) -> "PivotTopicNetworkGUI":  # pylint: disable=unused-argument
        wu.register_observer(self._single_pivot_key_picker, handler=self.update_handler, value=value)
        wu.register_observer(self._threshold, handler=self.update_handler, value=value)
        wu.register_observer(self._year_range, handler=self.update_handler, value=value)
        wu.register_observer(self._ignores, handler=self.update_handler, value=value)
        wu.register_observer(self._aggregate, handler=self.update_handler, value=value)
        wu.register_observer(self._threshold, handler=self.update_handler, value=value)
        wu.register_observer(self._grouped_threshold, handler=self.update_handler, value=value)
        wu.register_observer(self._scale, handler=self.display_handler, value=value)
        wu.register_observer(self._output_format, handler=self.display_handler, value=value)
        wu.register_observer(self._network_layout, handler=self.display_handler, value=value)
        return self

    def layout(self) -> w.VBox:
        extra_widgets: w.VBox = self.extra_widgets()

        return w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                w.HTML("<b>Pivot Key</b>"),
                                self._single_pivot_key_picker,
                                self._year_range_label,
                                self._year_range,
                                self._threshold_label,
                                self._threshold,
                                self._grouped_threshold_label,
                                self._grouped_threshold,
                            ]  # , layout={'border': '1px solid black'}
                        ),
                        w.VBox(
                            [
                                w.HTML("<b>Ignore topics</b>"),
                                w.VBox([self._ignores]),
                            ]
                        ),
                        self.default_pivot_keys_layout(vertical=True, rows=7, width='120px'),
                    ]
                    + ([extra_widgets] if extra_widgets else [])
                    + [
                        w.VBox(
                            [
                                self._scale_label,
                                self._scale,
                                w.HTML("<b>Output</b>"),
                                self._output_format,
                                w.HTML("<b>Network layout</b>"),
                                self._network_layout,
                            ]
                        ),
                        w.VBox(
                            [
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

        network_data: pd.DataFrame = (
            self.inferred_topics.calculator.reset()
            .filter_by_keys(**self.filter_opts.opts)
            .threshold(threshold=self.threshold)
            .filter_by_topics(topic_ids=self.ignores, negate=True)
            .overload(self.picked_pivot_id)
            .to_pivot_topic_network(
                pivot_key_id=self.picked_pivot_id,
                pivot_key_name=self.picked_pivot_name,
                pivot_key_map=self.picked_pivot_value_mapping,
                aggregate=self.aggregate,
                threshold=self.grouped_threshold,
            )
            .value
        )

        if len(network_data) == 0:
            raise pu.EmptyDataError()

        return network_data

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
                g = ntm.table_widget(self.network_data)
                display(g)
            else:

                plot_bipartite_dataframe(
                    data=self.network_data,
                    layout_algorithm=self._network_layout.value,
                    scale=self._scale.value,
                    titles=self.titles,
                    source_name="topic_id",
                    target_name=self.picked_pivot_name,
                    element_id=self.text_id,
                )

    @property
    def aggregate(self) -> str:
        return self._aggregate.value

    @property
    def threshold(self) -> str:
        """Threshodl for topic's weight in document"""
        return self._threshold.value

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        return pu.PropertyValueMaskingOpts(year=self.years).update(super().filter_opts)

    @property
    def grouped_threshold(self) -> str:
        """Threshold for grouped (aggregated) pivot-topic weights"""
        return self._grouped_threshold.value

    @property
    def years(self) -> tuple[int, int]:
        return self._year_range.value

    @property
    def output_format(self) -> str:
        return self._output_format.value.lower()

    @property
    def ignores(self) -> list[int]:
        return self._ignores.value
