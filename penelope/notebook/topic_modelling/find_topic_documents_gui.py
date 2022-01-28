from __future__ import annotations

import ipywidgets as w
import pandas as pd
from IPython.display import display

from penelope import topic_modelling as tm
from penelope import utility as pu
from penelope.notebook.widgets_utils import register_observer

from . import mixins as mx
from .model_container import TopicModelContainer


class FindTopicDocumentsGUI(mx.AlertMixIn, mx.TopicsStateGui):
    def __init__(self, state: TopicModelContainer | dict):
        super().__init__(state)

        timespan: tuple[int, int] = self.inferred_topics.year_period

        self._threshold: w.FloatSlider = w.FloatSlider(min=0.01, max=1.0, value=0.05, step=0.01)
        self._top_token_slider: w.IntSlider = w.IntSlider(min=3, max=200, value=3, disabled=False)
        self._max_n_top: w.IntSlider = w.IntSlider(min=1, max=50000, value=500, disabled=False)
        self._year_range: w.IntRangeSlider = w.IntRangeSlider(min=timespan[0], max=timespan[1], value=timespan)
        self._find_text: w.Text = w.Text(description="", layout={'width': '160px'})
        self._output: w.Output = w.Output()
        self._toplist_label: w.HTML = w.HTML("Tokens toplist threshold for token")
        self._extra_placeholder: w.Box = None
        self._compute: w.Button = w.Button(description='Show!', button_style='Success', layout={'width': '140px'})
        self._auto_compute: w.ToggleButton = w.ToggleButton(
            description="auto", icon='check', value=True, layout={'width': '140px'}
        )

    def setup(self) -> "FindTopicDocumentsGUI":
        self._compute.on_click(self.update_handler)
        register_observer(self.auto_compute, handler=self._auto_compute_handler, value=True)
        return self

    def layout(self) -> w.VBox:
        return w.VBox(
            (
                w.HBox(
                    [
                        w.VBox(
                            [
                                w.HTML("<b>Threshold</b> (topic's weight in doc)"),
                                self._threshold,
                                self._toplist_label,
                                self._top_token_slider,
                                w.HTML("<b>Max result count</b>"),
                                self._max_n_top,
                            ]
                        ),
                        w.VBox(
                            [
                                w.HTML("<b>Year range</b>"),
                                self._year_range,
                                w.HTML("<b>Filter topics by token</b>"),
                                self._find_text,
                            ]
                        ),
                    ]
                    + ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                    + [
                        w.VBox(
                            [
                                w.HTML("&nbsp;"),
                                self._auto_compute,
                                self._compute,
                                self._alert,
                            ]
                        )
                    ]
                ),
                self._output,
            )
        )

    def _find_text_handler(self, *_):
        self._top_token_slider.disabled = len(self._find_text.value) < 2

    def observe(self, value: bool) -> "FindTopicDocumentsGUI":
        value = value and self.auto_compute  # Never override autocompute
        register_observer(self._threshold.observe, handler=self.update_handler, value=value)
        register_observer(self._year_range.observe, handler=self.update_handler, value=value)
        register_observer(self._top_token_slider.observe, handler=self.update_handler, value=value)
        register_observer(self._find_text.observe, handler=self.update_handler, value=value)
        register_observer(self._find_text.observe, handler=self._find_text_handler, value=value)
        return self

    def _auto_compute_handler(self, *_):
        self._compute.disabled = self.auto_compute
        self._auto_compute.icon = 'check' if self.auto_compute else ''
        self.observe(self.auto_compute)
        if self.auto_compute:
            self.update_handler()

    @property
    def threshold(self) -> float:
        return self._threshold.value

    @property
    def years(self) -> tuple[int, int]:
        return self._year_range.value

    @property
    def text(self) -> str:
        return self._find_text.value

    @property
    def n_top_token(self) -> int:
        return self._top_token_slider.value

    @property
    def max_n_top(self) -> int:
        return self._max_n_top.value

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        return pu.PropertyValueMaskingOpts(year=self.years)

    @property
    def auto_compute(self) -> bool:
        return self._auto_compute.value

    @property
    def dtw_calculator(self) -> tm.DocumentTopicsCalculator:
        return self.inferred_topics.calculator

    def update(self) -> pd.DataFrame:
        if len(self.text) < 3:
            return None
        document_topics: pd.DataFrame = (
            self.dtw_calculator.reset()
            .filter_by_text(search_text=self.text, n_top=self.n_top_token)
            .filter_by_document_keys(**self.filter_opts.data)
            .threshold(self.threshold)
            .filter_by_n_top(self.max_n_top)
            .value
        )

        return document_topics

    def update_handler(self, *_):

        self._toplist_label.value = f"<b>Token must be within top {self._top_token_slider.value} topic tokens</b>"
        self._output.clear_output()

        with self._output:
            # try:

            if len(self.text) < 3:
                self.alert("Please enter a token with at least three chars.")
                return

            self.alert("Computing...")
            document_topics: pd.DataFrame = self.update()
            if document_topics is not None:
                display(document_topics)
            self.alert("✔️")
            # except Exception as ex:
            #     self.warn(str(ex))


def create_gui(state: TopicModelContainer) -> FindTopicDocumentsGUI:

    gui: FindTopicDocumentsGUI = FindTopicDocumentsGUI(state).setup()

    display(gui.layout())

    return gui
