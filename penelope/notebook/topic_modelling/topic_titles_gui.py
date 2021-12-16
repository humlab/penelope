import abc

import pandas as pd
from ipysheet import from_dataframe
from IPython.core.display import Javascript
from IPython.display import display as IPython_display
from ipywidgets import Button, HBox, IntSlider, Output, Text, VBox  # type: ignore
from penelope.notebook.utility import create_js_download
from penelope.topic_modelling import filter_topic_tokens_overview

pd.options.mode.chained_assignment = None


class TopicTitlesGUI(abc.ABC):
    def __init__(self):

        self.topics: pd.DataFrame = None
        self.reduced_topics: pd.DataFrame = None

        self.count_slider: IntSlider = IntSlider(
            description="Tokens",
            min=25,
            max=200,
            value=50,
            continuous_update=False,
        )
        self.search_text: Text = Text(description="Find")
        self.download_button: Button = Button(description="Download")
        self.output: Output = Output()
        self.js_download: Javascript = None

    def layout(self):
        return VBox((HBox((self.count_slider, self.search_text, self.download_button)), self.output))

    def download(self, *_):
        with self.output:
            js_download = create_js_download(self.reduced_topics, index=True)
            if js_download is not None:
                IPython_display(js_download)

    def _update(self, *_):
        self.output.clear_output()
        with self.output:
            self.reduce_topics()
            self.update()

    def reduce_topics(self):
        self.reduced_topics = filter_topic_tokens_overview(
            self.topics, search_text=self.search_text.value, n_top=self.count_slider.value
        )

    def update(self) -> None:
        IPython_display(self.reduced_topics)

    def display(self, topics: pd.DataFrame) -> "TopicTitlesGUI":

        self.topics = topics
        self.count_slider.observe(self._update, "value")
        self.search_text.observe(self._update, "value")
        self.download_button.on_click(self.download)

        IPython_display(self.layout())

        return self


class PandasTopicTitlesGUI(TopicTitlesGUI):

    PANDAS_TABLE_STYLE = [
        dict(
            selector="th",
            props=[
                ('font-size', '11px'),
                ('text-align', 'left'),
                ('font-weight', 'bold'),
                ('color', '#6d6d6d'),
                ('background-color', '#f7f7f9'),
            ],
        ),
        dict(
            selector="td",
            props=[
                ('font-size', '11px'),
                ('text-align', 'left'),
            ],
        ),
    ]

    def update(self, *_):
        styled_reduced_topics = self.reduced_topics.style.set_table_styles(self.PANDAS_TABLE_STYLE)
        IPython_display(styled_reduced_topics)

    def display(self, topics: pd.DataFrame) -> "TopicTitlesGUI":
        super().display(topics=topics)
        # pd.options.display.max_colwidth = None
        pd.set_option('colheader_justify', 'left')
        return self


class EditTopicTitlesGUI(PandasTopicTitlesGUI):  # pylint: disable=too-many-instance-attributes
    def __init__(self):
        super().__init__()
        self.save_button: Button = Button(description="Save")
        self.sheet = None

    def reduce_topics(self):
        ...

    def update(self, *_):
        ...
        # IPython_display(self.sheet)

    def display(self, topics: pd.DataFrame) -> "TopicTitlesGUI":
        super().display(topics)
        self.sheet = from_dataframe(self.topics)
        IPython_display(self.sheet)
        return self


def display_gui(topics: pd.DataFrame, displayer_cls: type = PandasTopicTitlesGUI):
    _ = displayer_cls().display(topics=topics)
