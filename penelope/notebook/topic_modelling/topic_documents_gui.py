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


# FIXME #157 Strange weights are displayed for certain topics (integer value)
class TopicDocumentsGUI(mx.AlertMixIn, mx.TopicsStateGui):
    WIDGET_HEIGHT: str = '450px'

    def __init__(self, state: TopicModelContainer | dict):
        super().__init__(state=state)
        slider_opts = {
            'continuous_update': False,
            'layout': dict(width='140px'),
            'readout': False,
            'handle_color': 'lightblue',
        }
        timespan: tuple[int, int] = self.inferred_topics.year_period
        yearspan: tuple[int, int] = self.inferred_topics.startspan(10)

        self._threshold_label: w.HTML = w.HTML("<b>Threshold</b>")
        self._threshold: w.FloatSlider = w.FloatSlider(min=0.01, max=1.0, value=0.05, step=0.01, **slider_opts)
        self._max_count_label: w.HTML = w.HTML("<b>Max result count</b>")
        self._max_count: w.IntSlider = w.IntSlider(min=1, max=50000, value=500, **slider_opts)
        self._year_range_label: w.HTML = w.HTML("Years")
        self._year_range: w.IntRangeSlider = w.IntRangeSlider(
            min=timespan[0], max=timespan[1], step=1, value=yearspan, **slider_opts
        )
        self._output: w.Output = w.Output(layout={'width': '99%'})
        self._extra_placeholder: w.Box = None
        self._content_placeholder: w.Box = None
        self._compute: w.Button = w.Button(description='Show!', button_style='Success', layout={'width': '80px'})
        self._auto_compute: w.ToggleButton = w.ToggleButton(description="auto", value=False, layout={'width': '80px'})
        self._table_widget: TableWidget = None
        self._document_click_handler: Callable[[pd.Series, Any], None] = None

    def setup(self, **kwargs) -> "TopicDocumentsGUI":  # pylint: disable=arguments-differ,unused-argument
        self._compute.on_click(self.update_handler)
        self.observe_slider_update_label(self._year_range, self._year_range_label, "Years")
        self.observe_slider_update_label(self._threshold, self._threshold_label, "Threshold")
        self.observe_slider_update_label(self._max_count, self._max_count_label, "Max result count")
        wu.register_observer(self._auto_compute, handler=self._auto_compute_handler, value=True)
        return self

    def layout(self) -> w.Widget:
        return None

    def _auto_compute_handler(self, *_):
        self._auto_compute.icon = 'check' if self.auto_compute else ''
        self._compute.disabled = self.auto_compute
        self.observe(value=self.auto_compute, handler=self.update_handler)
        if self.auto_compute:
            self.update_handler()

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
                    self._table_widget.layout.height = self.WIDGET_HEIGHT
                    display(self._table_widget)

                self.alert("✅")
            except Exception as ex:
                self.warn(str(ex))

    def update(self) -> pd.DataFrame:
        raise NotImplementedError("base class")


class BrowseTopicDocumentsGUI(mx.NextPrevTopicMixIn, TopicDocumentsGUI):
    def __init__(self, state: TopicModelContainer | dict):
        super().__init__(state=state)

        self._text: w.HTML = w.HTML()

    def setup(self, **kwargs) -> "BrowseTopicDocumentsGUI":  # pylint: disable=arguments-differ
        super().setup(**kwargs)
        self.topic_id = (0, self.inferred_n_topics - 1, self.inferred_topics.topic_labels)
        self._topic_id.observe(self.update_handler, names='value')
        self._threshold.observe(self.update_handler, names='value')
        self._max_count.observe(self.update_handler, names='value')

        return self

    def layout(self) -> w.Widget:
        _output_container = w.VBox([self._text, self._output])

        _layout = w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                self._threshold_label,
                                self._threshold,
                                self._year_range_label,
                                self._year_range,
                            ]
                        ),
                    ]
                    + [
                        w.VBox(
                            [
                                self._max_count_label,
                                self._max_count,
                                self._next_prev_layout,
                            ]
                        )
                    ]
                    + ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                ),
                w.HBox([w.HTML("&nbsp;"), self._auto_compute, self._compute, self._alert]),
                _output_container,
            ],
            layout={'width': '50%'},
        )
        if self._content_placeholder is not None:
            self._content_placeholder.layout.width = '50%'
            _layout = w.HBox([_layout, self._content_placeholder], width='100%')

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
        self._find_text.layout.width = '160px'
        self.observe_slider_update_label(self._n_top_token, self._n_top_token_label, "Toplist threshold")

    def layout(self) -> w.VBox:
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
                                self._max_count_label,
                                self._max_count,
                            ]
                        ),
                        w.VBox(
                            [
                                self._year_range_label,
                                self._year_range,
                                w.HTML("<b>Filter topics by token</b>"),
                                self._find_text,
                            ]
                        ),
                    ]
                    + ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                ),
                w.HBox([w.HTML("&nbsp;"), self._auto_compute, self._compute, self._alert]),
                self._output,
            ],
            layout={'width': '50%'},
        )
        if self._content_placeholder is not None:
            self._content_placeholder.layout.width = '50%'
            _layout = w.HBox([_layout, self._content_placeholder], width='100%')

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
        print(self.filter_opts.opts)
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
                layout={'width': opts.get('width', '200px')},
                rows=opts.get("rows", 8),
            )

        def setup(self, **kwargs):
            if self.config and self.config.pivot_keys:
                self.pivot_keys = self.config.pivot_keys

            return super().setup(**kwargs)

        # def update(self) -> pd.DataFrame:
        #     _ = super().update()
        #     """note: at this point dtw is equal to calculator.data"""
        #     self.alert("preparing data, please wait...")
        #     calculator: tx.DocumentTopicsCalculator = self.inferred_topics.calculator
        #     data: pd.DataFrame = self.person_codecs.decode(
        #         calculator.overload(includes="protocol_name,document_name,gender_id,party_id,person_id").value,
        #         drop=True,
        #     )
        #     self.alert("Done!")
        #     return data

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
