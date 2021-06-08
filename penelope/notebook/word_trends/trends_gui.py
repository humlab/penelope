from typing import Sequence

from ipywidgets import (
    BoundedIntText,
    Dropdown,
    HBox,
    Label,
    Layout,
    Output,
    SelectMultiple,
    Tab,
    Textarea,
    ToggleButton,
    VBox,
)
from penelope.common.keyness import KeynessMetric
from penelope.utility import get_logger

from .displayers import ITrendDisplayer
from .interface import TrendsData, TrendsOpts

logger = get_logger()

BUTTON_LAYOUT = Layout(width='120px')
OUTPUT_LAYOUT = Layout(width='600px')


class TrendsGUI:
    """GUI component that displays word trends"""

    def __init__(self, n_top_count: int = 5000):
        self.trends_data: TrendsData = None

        self._tab: Tab = Tab(layout={'width': '98%'})
        self._picker: SelectMultiple = SelectMultiple(
            description="", options=[], value=[], rows=30, layout={'width': '250px'}
        )
        self._normalize: ToggleButton = ToggleButton(
            description="Normalize", icon='check', value=False, layout=BUTTON_LAYOUT
        )
        self._keyness: ToggleButton = Dropdown(
            description="",
            value=KeynessMetric.TF,
            options={"TF": KeynessMetric.TF, "TF-IDF": KeynessMetric.TF_IDF},
            layout=BUTTON_LAYOUT,
        )
        self._smooth: ToggleButton = ToggleButton(description="Smooth", icon='check', value=False, layout=BUTTON_LAYOUT)
        self._group_by: Dropdown = Dropdown(
            options=['year', 'lustrum', 'decade'],
            value='year',
            description='',
            disabled=False,
            layout=Layout(width='75px'),
        )
        self._status: Label = Label(layout=Layout(width='50%', border="0px transparent white"))
        self._words: Textarea = Textarea(
            description="",
            rows=2,
            value="",
            placeholder='Enter words, wildcards and/or regexps such as "information", "info*", "*ment",  "|.*tion$|"',
            layout=Layout(width='98%'),
        )
        self._word_count: BoundedIntText = BoundedIntText(
            value=n_top_count,
            min=3,
            max=100000,
            step=10,
            description='Max words:',
            disabled=False,
            layout={'width': '180px'},
        )
        self._displayers: Sequence[ITrendDisplayer] = []

    def layout(self) -> VBox:
        return VBox(
            [
                HBox(
                    [
                        self._keyness,
                        self._normalize,
                        self._smooth,
                        self._group_by,
                        self._word_count,
                        self._status,
                    ]
                ),
                self._words,
                HBox([self._picker, self._tab], layout={'width': '98%'}),
            ]
        )

    def _plot_trends(self, *_):

        try:
            if self.trends_data is None:
                self.alert("😮 Please load a corpus (no trends data) !")
                return

            if self.trends_data is None or self.trends_data.corpus is None:
                self.alert("😥 Please load a corpus (no corpus in trends data) !")
                return

            corpus = self.trends_data.transform(self.options).transformed_corpus

            self.current_displayer.display(
                corpus=corpus,
                indices=corpus.token_indices(self._picker.value),
                smooth=self.smooth,
                category_name=self.trends_data.category_column,
            )

            self.alert("✔")

        except ValueError as ex:
            self.alert(str(ex))
        except Exception as ex:
            logger.exception(ex)
            self.warn(str(ex))
            raise

    def _update_picker(self, *_):

        _words = self.trends_data.find_words(self.options)
        _values = [w for w in self._picker.value if w in _words]

        self._picker.value = []
        self._picker.options = _words
        self._picker.value = _values

        if len(_words) == 0:
            self.alert("😢 Found no matching words!")
        else:
            self.alert(
                f"✔ Displaying {len(_words)} matching tokens. {'' if len(_words) < self.word_count else ' (result truncated)'}"
            )

    def setup(self, *, displayers: Sequence[ITrendDisplayer]) -> "TrendsGUI":

        for i, cls in enumerate(displayers):
            displayer: ITrendDisplayer = cls()
            self._displayers.append(displayer)
            displayer.output = Output()
            with displayer.output:
                displayer.setup()

        self._tab.children = [d.output for d in self._displayers]
        for i, d in enumerate(self._displayers):
            self._tab.set_title(i, d.name)

        self._words.observe(self._update_picker, names='value')
        self._tab.observe(self._plot_trends, 'selected_index')
        self._normalize.observe(self._plot_trends, names='value')
        self._keyness.observe(self._plot_trends, names='value')
        self._smooth.observe(self._plot_trends, names='value')
        self._group_by.observe(self._plot_trends, names='value')
        self._picker.observe(self._plot_trends, names='value')

        return self

    def display(self, *, trends_data: TrendsData):
        if trends_data is None:
            raise ValueError("No trends data supplied!")
        if trends_data.corpus is None:
            raise ValueError("No corpus supplied!")
        if self._picker is not None:
            self._picker.values = []
            self._picker.options = []
        self.trends_data = trends_data
        self._plot_trends()

    @property
    def current_displayer(self) -> ITrendDisplayer:
        return self._displayers[self._tab.selected_index]

    @property
    def current_output(self):
        return self.current_displayer.output

    def alert(self, msg: str):
        self._status.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")

    @property
    def words_or_regexp(self):
        return ' '.join(self._words.value.split()).split()

    @property
    def words(self):
        return self._picker.value

    @property
    def smooth(self) -> bool:
        return self._smooth.value

    @property
    def keyness(self) -> KeynessMetric:
        return self._keyness.value

    @property
    def normalize(self) -> bool:
        return self._normalize.value

    @property
    def group_by(self) -> str:
        return self._group_by.value

    @property
    def word_count(self) -> int:
        return self._word_count.value

    @property
    def options(self) -> TrendsOpts:
        return TrendsOpts(
            normalize=self.normalize,
            smooth=self.smooth,
            keyness=self.keyness,
            group_by=self.group_by,
            word_count=self.word_count,
            words=self.words_or_regexp,
            descending=True,
        )
