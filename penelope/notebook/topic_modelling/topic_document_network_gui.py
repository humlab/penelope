from __future__ import annotations

import uuid
from typing import Any, Callable

import ipywidgets as w
import pandas as pd
from IPython.display import display

from penelope import network as nx
from penelope import utility as pu

from .. import grid_utility as gu
from .. import mixins as ox
from .. import topic_modelling as ntm
from .. import widgets_utils as wu
from . import mixins as mx

LAYOUT_OPTIONS = ['Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
OUTPUT_OPTIONS = {'Network': 'network', 'Table': 'table', 'Excel': 'XLSX', 'CSV': 'CSV', 'Clipboard': 'clipboard'}


# pylint: disable=too-many-locals,too-many-arguments,too-many-instance-attributes,disable=arguments-differ


class TopicDocumentNetworkGui(ox.PivotKeysMixIn, mx.AlertMixIn, mx.ComputeMixIn, ntm.TopicsStateGui):
    def __init__(self, pivot_key_specs: ox.PivotKeySpecArg, state: ntm.TopicModelContainer, **kwargs):
        super().__init__(pivot_key_specs, state=state, **kwargs)

        slider_opts: dict = {
            'continuous_update': False,
            'layout': dict(width='140px'),
            'readout': False,
            'handle_color': 'lightblue',
        }

        self.text_id: str = f"{str(uuid.uuid4())[:6]}"
        self.network_data: pd.DataFrame = None
        self.topic_proportions: pd.DataFrame = None
        self.titles: pd.DataFrame = None
        self.topics_ids_header: str = "<b>Ignore topics</b>"
        self.default_threshold: float = 0.5

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
        self._threshold: w.FloatSlider = w.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.10, **slider_opts)
        self._network_layout: w.Dropdown = w.Dropdown(
            options=LAYOUT_OPTIONS, value='Fruchterman-Reingold', layout=dict(width='140px')
        )
        self._output_format: w.Dropdown = w.Dropdown(
            description='', options=OUTPUT_OPTIONS, value='network', layout=dict(width='140px')
        )
        self._output: w.Output = w.Output()
        self._content_placeholder: w.Box = None
        self._extra_placeholder: w.Box = w.VBox()
        self._topic_ids: w.SelectMultiple = w.SelectMultiple(
            options=self.topic_id_options(), value=[], rows=8, layout=dict(width='180px')
        )

    def topic_id_options(self) -> list[tuple[str, int]]:
        return [("", None)] + super().topic_id_options()

    def setup(self, default_threshold: float = None, **kwargs) -> "TopicDocumentNetworkGui":
        super().setup(**kwargs)

        self._compute_handler: Callable[[Any], None] = self.update_handler
        self.topic_proportions: pd.DataFrame = self.inferred_topics.calculator.reset().topic_proportions()
        self.titles: pd.DataFrame = self.inferred_topics.get_topic_titles()
        self._threshold.value = default_threshold or self.default_threshold

        self.observe_slider_update_label(self._year_range, self._year_range_label, "Years")
        self.observe_slider_update_label(self._threshold, self._threshold_label, "Threshold")
        self.observe_slider_update_label(self._scale, self._scale_label, "Scale")
        self.observe(value=True, handler=self.update_handler)
        self.inferred_topics.calculator.reset()
        return self

    def observe(self, value: bool, **kwargs):  # pylint: disable=unused-argument
        # wu.register_observer(self._single_pivot_key_picker, handler=self.update_handler, value=value)
        # wu.register_observer(self._threshold, handler=self.update_handler, value=value)
        # wu.register_observer(self._year_range, handler=self.update_handler, value=value)
        # wu.register_observer(self._topic_ids, handler=self.update_handler, value=value)
        # wu.register_observer(self._threshold, handler=self.update_handler, value=value)
        wu.register_observer(self._scale, handler=self.display_handler, value=value)
        wu.register_observer(self._output_format, handler=self.display_handler, value=value)
        wu.register_observer(self._network_layout, handler=self.display_handler, value=value)

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
                            ]
                        ),
                        w.VBox(
                            [
                                w.HTML(self.topics_ids_header),
                                self._topic_ids,
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
                        w.VBox([self._auto_compute, self._compute, self._alert]),
                    ]
                ),
                self._output,
                w.HBox([self._text] + ([self._content_placeholder] if self._content_placeholder is not None else [])),
            ]
        )

    def extra_widgets(self) -> w.VBox:
        return self._extra_placeholder

    def compute(self) -> pd.DataFrame:

        return (
            self.inferred_topics.calculator.reset()
            .filter_by_keys(**self.filter_opts.opts)
            .threshold(threshold=self.threshold)
            .filter_by_topics(topic_ids=self.topic_ids, negate=True)
        ).value

    def update(self) -> pd.DataFrame:

        network_data: pd.DataFrame = self.compute()
        network_data["weight"] = pu.clamp_values(list(network_data["weight"]), (0.1, 2.0))
        di: pd.DataFrame = self.inferred_topics.document_index.pipe(pu.set_index, columns='document_id')[
            ["document_name"]
        ]
        network_data = network_data.pipe(pu.set_index, columns='document_id').merge(
            di, left_index=True, right_index=True
        )
        network_data["title"] = network_data["document_name"]

        if self.topic_labels is not None:
            network_data['topic_id'] = network_data['topic_id'].apply(self.topic_labels.get)

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

            elif self.output_format in ('xlsx', 'csv', 'clipboard', 'table', 'gephi'):

                data: pd.DataFrame = self.network_data

                if self.output_format == "gephi":
                    data = data[['topic_id', "title", 'weight']]
                    data.columns = ['Source', 'Target', 'Weight']

                if self.output_format != "table":
                    pu.ts_store(data=data, extension=self.output_format, basename='heatmap_weights')

                g: gu.TableWidget = gu.table_widget(data)
                display(g)
            else:
                nx.plot_highlighted_bipartite_dataframe(
                    network_data=self.network_data,
                    network_layout=self.network_layout,
                    highlight_topic_ids=self.highlight_ids,
                    titles=self.titles,
                    scale=self.scale,
                    source_name="title",  # FIXME:self.picked_pivot_name
                    target_name="topic_id",
                    element_id=self.text_id,
                )

    @property
    def threshold(self) -> str:
        """Threshodl for topic's weight in document"""
        return self._threshold.value

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        return pu.PropertyValueMaskingOpts(year=self.years).update(super().filter_opts)

    @property
    def years(self) -> tuple[int, int]:
        return self._year_range.value

    @property
    def network_layout(self) -> str:
        return self._network_layout.value

    @property
    def scale(self) -> float:
        return self._scale.value

    @property
    def output_format(self) -> str:
        return self._output_format.value.lower()

    @property
    def topic_ids(self) -> list[str]:
        return self._topic_ids.value

    @property
    def highlight_ids(self) -> list[str]:
        return None


class DefaultTopicDocumentNetworkGui(TopicDocumentNetworkGui):
    ...


class FocusTopicDocumentNetworkGui(TopicDocumentNetworkGui):
    def __init__(self, pivot_key_specs: ox.PivotKeySpecArg, state: ntm.TopicModelContainer, **kwargs):
        super().__init__(pivot_key_specs, state, **kwargs)
        self.topics_ids_header: str = "<b>Focus topics</b>"
        self.default_threshold: float = 0.1

    def compute(self) -> pd.DataFrame:
        return (
            self.inferred_topics.calculator.reset()
            .filter_by_keys(**self.filter_opts.opts)
            .threshold(threshold=self.threshold)
            .filter_by_focus_topics(topic_ids=self.topic_ids)
        ).value

    @property
    def highlight_ids(self) -> list[str]:
        return self.topic_ids
