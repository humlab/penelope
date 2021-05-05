from typing import Any

from IPython.display import display as ip_display
from ipywidgets import HTML, FloatSlider, IntSlider, Output, Text, VBox
from ipywidgets.widgets.widget_box import HBox

from .display_topic_titles import reduce_topic_tokens_overview


class GUI:
    def __init__(self):
        self.threshold_slider: FloatSlider = FloatSlider(min=0.01, max=1.0, value=0.2)
        self.top_token_slider: IntSlider = IntSlider(min=3, max=200, value=3, disabled=True)
        self.find_text: Text = Text(description="")
        self.output: Output = Output()
        self.callback = lambda *_: ()
        self.toplist_label: HTML = HTML("Tokens toplist threshold for token")

    def layout(self):
        return VBox(
            (
                HBox(
                    (
                        VBox(
                            (
                                HTML("Topic weight in document threshold"),
                                self.threshold_slider,
                            )
                        ),
                        VBox(
                            (
                                HTML("<b>Find topics containing token</b>"),
                                self.find_text,
                            )
                        ),
                        VBox(
                            (
                                self.toplist_label,
                                self.top_token_slider,
                            )
                        ),
                    )
                ),
                self.output,
            )
        )

    def _callback(self, *_):
        self.toplist_label.value = f"<b>Token must be within top {self.top_token_slider.value} topic tokens</b>"
        self.callback(
            gui=self,
        )

    def _find_text(self, *_):
        self.top_token_slider.disabled = len(self.find_text.value) < 2

    def setup(self, callback):
        self.threshold_slider.observe(self._callback, 'value')
        self.top_token_slider.observe(self._callback, 'value')
        self.find_text.observe(self._callback, 'value')
        self.find_text.observe(self._find_text, 'value')
        self.callback = callback or self.callback
        return self

    @property
    def threshold(self) -> float:
        return self.threshold_slider.value

    @property
    def text(self) -> str:
        return self.find_text.value

    @property
    def top(self) -> int:
        return self.top_token_slider.value


def gui_controller(document_topic_weights, topic_token_overview):
    def display_document_topic_weights(gui: Any):
        gui.output.clear_output()
        with gui.output:

            df = document_topic_weights

            if len(gui.text) > 2:

                topic_ids = reduce_topic_tokens_overview(
                    topic_token_overview,
                    gui.top,
                    gui.text,
                ).index.tolist()

                df = df[df.topic_id.isin(topic_ids)]

            df = df[df.weight >= gui.threshold]

            ip_display(df)

    gui = GUI().setup(callback=display_document_topic_weights)

    ip_display(gui.layout())
