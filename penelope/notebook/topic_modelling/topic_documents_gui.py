from __future__ import annotations

from typing import Any, Callable

import ipywidgets as w
import pandas as pd
from IPython.display import display

from penelope import utility as pu
from penelope.notebook import widgets_utils as wu

from .. import mixins as nx
from ..grid_utility import TableWidget, table_widget
from . import mixins as mx
from .model_container import TopicModelContainer


class NotLoadedError(Exception):
    ...


# FIXME #157 Strange weights are displayed for certain topics (integer value)
class TopicDocumentsGUI(mx.ComputeMixIn, mx.AlertMixIn, mx.TopicsStateGui):
    def __init__(self, state: TopicModelContainer | dict):
        if state.inferred_topics is None:
            raise NotLoadedError("Topic model has not been loaded") from None

        super().__init__(state=state)

        self.table_widget_height: str = '450px'

        slider_opts = {
            'continuous_update': False,
            'layout': dict(width='140px'),
            'readout': False,
            'handle_color': 'lightblue',
        }
        timespan: tuple[int, int] = self.inferred_topics.year_period
        yearspan: tuple[int, int] = self.inferred_topics.startspan(10)

        self._threshold_label: w.HTML = w.HTML("<b>Threshold</b>")
        self._threshold: w.FloatSlider = w.FloatSlider(
            min=0.01,
            max=1.0,
            value=0.05,
            step=0.01,
            tooltop="Filter out documents if matching topic's threshold is lower than this value",
            **slider_opts,
        )
        self._max_count_label: w.HTML = w.HTML("<b>Max result count</b>")
        self._max_count: w.IntSlider = w.IntSlider(
            min=1,
            max=50000,
            value=500,
            tooltip="Maximum number of documents to display (sorted by weight descending)",
            **slider_opts,
        )
        self._year_range_label: w.HTML = w.HTML("Years")
        self._year_range: w.IntRangeSlider = w.IntRangeSlider(
            min=timespan[0],
            max=timespan[1],
            step=1,
            value=yearspan,
            tooltip="Ignore documents outside this period.",
            **slider_opts,
        )
        self._output: w.Output = w.Output(layout={'width': '100%'})
        self._extra_placeholder: w.Box = None
        self._content_placeholder: w.Box = None
        self._table_widget: TableWidget = None
        self._document_click_handler: Callable[[pd.Series, Any], None] = None

    def setup(self, **kwargs) -> "TopicDocumentsGUI":  # pylint: disable=arguments-differ,unused-argument
        self._compute_handler = self.update_handler

        super().setup(**kwargs)

        self.observe_slider_update_label(self._year_range, self._year_range_label, "Years")
        self.observe_slider_update_label(self._threshold, self._threshold_label, "Threshold")
        self.observe_slider_update_label(self._max_count, self._max_count_label, "Max result count")
        return self

    def layout(self) -> w.Widget:
        return None

    @property
    def threshold(self) -> float:
        return self._threshold.value

    @property
    def years(self) -> tuple[int, int]:
        return self._year_range.value

    @property
    def max_count(self) -> int:
        return self._max_count.value

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        return pu.PropertyValueMaskingOpts(year=self.years)

    @property
    def auto_compute(self) -> bool:
        return self._auto_compute.value

    def observe(self, value: bool, **kwargs) -> "TopicDocumentsGUI":
        if hasattr(super(), "observe"):
            getattr(super(), "observe")(value=value, handler=self.update_handler, **kwargs)

        value = value and self.auto_compute  # Never override autocompute
        wu.register_observer(self._threshold, handler=self.update_handler, value=value)
        wu.register_observer(self._year_range, handler=self.update_handler, value=value)
        return self

    def update_handler(self, *_):
        self._output.clear_output()

        with self._output:
            try:
                self.alert("Computing...")
                data: pd.DataFrame = self.update()

                if data is not None:
                    self._table_widget: TableWidget = table_widget(data, handler=self._document_click_handler)
                    self._table_widget.layout.height = self.table_widget_height
                    display(self._table_widget)

                self.alert("✅")
            except Exception as ex:
                self.warn(str(ex))

    def update(self) -> pd.DataFrame:
        raise NotImplementedError("base class")


class BrowseTopicDocumentsGUI(mx.NextPrevTopicMixIn, TopicDocumentsGUI):
    def __init__(self, state: TopicModelContainer | dict):
        super().__init__(state=state)

        self._text: w.HTML = w.HTML(layout={'width': '100%'})

    def setup(self, **kwargs) -> "BrowseTopicDocumentsGUI":  # pylint: disable=arguments-differ
        super().setup(**kwargs)
        self.topic_id = (0, self.inferred_n_topics - 1, self.inferred_topics.topic_labels)
        self._topic_id.observe(self.update_handler, names='value')
        self._threshold.observe(self.update_handler, names='value')
        self._max_count.observe(self.update_handler, names='value')

        return self

    def layout(self) -> w.Widget:
        self._output.layout.width = '95%'
        _output: w.VBox = w.VBox([w.HTML("<h3>Documents:</h3><hr>", layout={'width': '95%'}), self._output])

        _content = [self._output]
        if self._content_placeholder is not None:
            _content = [_output, self._content_placeholder]
            self._content_placeholder.layout.width = '55%'
            _output.layout.width = '45%'
        else:
            _output.layout.width = '100%'

        _layout = w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                self._threshold_label,
                                self._threshold,
                            ]
                        ),
                        w.VBox(
                            [
                                self._year_range_label,
                                self._year_range,
                                self._max_count_label,
                                self._max_count,
                            ]
                        ),
                    ]
                    + ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                    + [
                        w.VBox(
                            [
                                self._alert,
                                self._next_prev_layout,
                                self.compute_default_layout,
                            ]
                        )
                    ]
                ),
                self._text,
                w.HBox(_content, layout={'width': '100%'}),
            ],
            layout={'width': '100%'},
        )

        return _layout

    def update(self) -> pd.DataFrame:
        data: pd.DataFrame = (
            self.inferred_topics.calculator.reset()
            .filter_by_data_keys(topic_id=self.topic_id)
            .threshold(self.threshold)
            .filter_by_document_keys(**self.filter_opts.opts)
            .filter_by_n_top(self.max_count)
            .overload("document_name,n_raw_tokens,n_tokens")
            .value
        )
        return data

    def update_handler(self, *_):
        self._text.value = self.inferred_topics.get_topic_title2(self.topic_id)
        super().update_handler()


class FindTopicDocumentsGUI(TopicDocumentsGUI):
    def __init__(self, state: TopicModelContainer | dict):
        super().__init__(state=state)

        self._n_top_token_label: w.HTML = w.HTML()
        self._n_top_token: w.IntSlider = w.IntSlider(min=3, max=200, value=3, readout=False)
        self._find_text: w.Text = w.Text(width='140px')
        self._n_top_token.layout.width = '140px'
        self._find_text.layout.width = '140px'
        self.observe_slider_update_label(self._n_top_token, self._n_top_token_label, "Toplist threshold")

    def layout(self) -> w.VBox:
        self._output.layout.width = '95%'
        _output: w.VBox = w.VBox([w.HTML("<h3>Documents:</h3><hr>", layout={'width': '95%'}), self._output])

        _content = [self._output]
        if self._content_placeholder is not None:
            _content = [_output, self._content_placeholder]
            self._content_placeholder.layout.width = '55%'
            _output.layout.width = '45%'
        else:
            _output.layout.width = '100%'

        _layout = w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                self._threshold_label,
                                self._threshold,
                                self._n_top_token_label,
                                self._n_top_token,
                            ]
                        ),
                        w.VBox(
                            [
                                self._year_range_label,
                                self._year_range,
                                self._max_count_label,
                                self._max_count,
                            ]
                        ),
                    ]
                    + ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                    + [
                        w.VBox(
                            [
                                w.HTML("<b>Filter topics by token</b>"),
                                self._find_text,
                                self.compute_default_layout,
                                self._alert,
                            ]
                        ),
                    ],
                    layout={'width': '100%'},
                ),
                w.HBox(_content, layout={'width': '100%'}),
            ],
            layout={'width': '100%'},
        )

        return _layout

    def _find_text_handler(self, *_):
        self._n_top_token.disabled = len(self._find_text.value) < 2

    def observe(self, value: bool, **kwargs) -> "FindTopicDocumentsGUI":
        super().observe(value=value, **kwargs)
        value = value and self.auto_compute  # Never override autocompute
        wu.register_observer(self._n_top_token, handler=self.update_handler, value=value)
        wu.register_observer(self._find_text, handler=self.update_handler, value=value)
        wu.register_observer(self._find_text, handler=self._find_text_handler, value=value)
        return self

    @property
    def text(self) -> str:
        return self._find_text.value

    @property
    def n_top_token(self) -> int:
        return self._n_top_token.value

    def update(self) -> pd.DataFrame:
        data: pd.DataFrame = (
            self.inferred_topics.calculator.reset()
            .filter_by_text(search_text=self.text, n_top=self.n_top_token)
            .threshold(self.threshold)
            .filter_by_document_keys(**self.filter_opts.opts)
            .filter_by_n_top(self.max_count)
            .overload("document_name,n_raw_tokens,n_tokens")
            .value
        )
        return data

    def update_handler(self, *_):
        if len(self.text) < 3:
            self.alert("Please enter a token with at least three characters.")
            return

        super().update_handler()


class WithPivotKeysText:
    class BrowseTopicDocumentsGUI(nx.TextRepositoryMixIn, nx.PivotKeysMixIn, BrowseTopicDocumentsGUI):
        def __init__(self, state: TopicModelContainer | dict, **opts):
            super().__init__(pivot_key_specs=None, state=state)
            self._opts = opts

            self._threshold.value = opts.get("threshold", 0.20)
            self._year_range.value = opts.get("year_span", (1990, 1992))
            self._extra_placeholder = self.default_pivot_keys_layout(
                vertical=opts.get("vertical", False),
                width=opts.get('width', '200px'),
                rows=opts.get("rows", 8),
            )

        def setup(self, **kwargs):
            if self.config and self.config.pivot_keys:
                self.pivot_keys = self.config.pivot_keys

            return super().setup(**kwargs)

        @property
        def filter_opts(self) -> pu.PropertyValueMaskingOpts:
            return super().filter_opts

    class FindTopicDocumentsGUI(nx.TextRepositoryMixIn, nx.PivotKeysMixIn, FindTopicDocumentsGUI):
        def __init__(self, state: TopicModelContainer | dict, **opts):
            super().__init__(pivot_key_specs=None, state=state)
            self._opts = opts
            self._threshold.value = opts.get("threshold", 0.20)
            self._extra_placeholder = self.default_pivot_keys_layout(
                vertical=opts.get("vertical", False),
                layout={'width': opts.get('width', '200px')},
                rows=opts.get("rows", 8),
            )

        def setup(self, **kwargs):
            if self.config and self.config.pivot_keys:
                self.pivot_keys = self.config.pivot_keys

            return super().setup(**kwargs)

        @property
        def filter_opts(self) -> pu.PropertyValueMaskingOpts:
            return super().filter_opts
