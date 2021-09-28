import abc
from typing import Sequence

from ipywidgets import (
    HTML,
    BoundedIntText,
    Button,
    Dropdown,
    GridBox,
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
from penelope.co_occurrence.bundle import Bundle
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.utility import get_logger

from .displayers import ITrendDisplayer
from .interface import TrendsComputeOpts, TrendsData

logger = get_logger()

BUTTON_LAYOUT = Layout(width='120px')
OUTPUT_LAYOUT = Layout(width='600px')


class TrendsBaseGUI(abc.ABC):
    """GUI component that displays word trends"""

    @abc.abstractmethod
    def keyness_widget(self) -> Dropdown:
        ...

    def __init__(self, n_top_count: int = 1000):
        self.trends_data: TrendsData = None

        self._tab: Tab = Tab(layout={'width': '98%'})
        self._picker: SelectMultiple = SelectMultiple(
            description="", options=[], value=[], rows=30, layout={'width': '250px'}
        )
        self._normalize: ToggleButton = ToggleButton(
            description="Normalize",
            icon='check',
            value=False,
            layout=BUTTON_LAYOUT,
            tooltip="Normalize entire corpus by selected period",
        )
        self._keyness: ToggleButton = self.keyness_widget()
        self._placeholder: VBox = VBox()
        self._smooth: ToggleButton = ToggleButton(description="Smooth", icon='check', value=False, layout=BUTTON_LAYOUT)
        self._time_period: Dropdown = Dropdown(
            options=['year', 'lustrum', 'decade'],
            value='decade',
            description='',
            disabled=False,
            layout=Layout(width='75px'),
        )
        self._alert: Label = Label(layout=Layout(width='auto', border="0px transparent white"))
        self._words: Textarea = Textarea(
            description="",
            rows=2,
            value="",
            placeholder='Enter words, wildcards and/or regexps such as "information", "info*", "*ment",  "|.*tion$|"',
            layout=Layout(width='98%'),
        )
        self._top_count: BoundedIntText = BoundedIntText(
            value=n_top_count,
            min=3,
            max=100000,
            step=10,
            description='',
            disabled=False,
            layout={'width': '180px'},
        )
        self._compute: Button = Button(description="Compute", button_style='success', layout=BUTTON_LAYOUT)
        self._displayers: Sequence[ITrendDisplayer] = []

    def _invalidate(self, value: bool = True):
        self._compute.disabled = not value
        self._words.disabled = value
        if value:
            for displayer in self._displayers:
                displayer.clear()
            self.alert(" 🗑 Data invalidated (press compute to update).")

    def _invalidate_handler(self, *_):
        self._invalidate(True)

    def layout(self) -> GridBox:
        layout: GridBox = GridBox(
            [
                HBox(
                    [
                        VBox([HTML("<b>Keyness</b>"), self._keyness]),
                        self._placeholder,
                        VBox([HTML("<b>Top count</b>"), self._top_count]),
                        VBox([HTML("<b>Grouping</b>"), self._time_period]),
                        VBox([self._normalize, self._smooth]),
                        VBox([HTML("⚽"), self._compute]),
                        VBox([HTML("📌"), self._alert]),
                    ],
                    layout={'width': '98%'},
                ),
                self._words,
                HBox([self._picker, self._tab], layout={'width': '98%'}),
            ],
            layout=Layout(width='auto'),
        )

        return layout

    def _compute_keyness(self, *_):
        try:
            self.buzy(True)

            if self.trends_data is None:
                self.alert("😮 Please load a corpus (no trends data) !")
                return

            if self.trends_data is None or self.trends_data.corpus is None:
                self.alert("😥 Please load a corpus (no corpus in trends data) !")
                return

            self.alert("⌛ Computing...")

            self.trends_data.transform(self.options)

            self.alert("✔")
            self._invalidate(False)

            self._plot_trends()
        except ValueError as ex:
            self.alert(str(ex))
        except Exception as ex:
            logger.exception(ex)
            self.warn(str(ex))
            raise
        self.buzy(False)

    def _plot_trends(self, *_):

        try:
            self.buzy(True)

            if self.trends_data is None or self.trends_data.transformed_corpus is None:
                self.alert("🥱 (not computed)")
                return

            self.alert("⌛ Preparing display...")
            self.current_displayer.display(
                corpus=self.trends_data.transformed_corpus,
                indices=self.trends_data.transformed_corpus.token_indices(self._picker.value),
                smooth=self.smooth,
                category_name=self.trends_data.category_column,
            )

            self.alert("🙂")

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
            self.alert("😫 Found no matching words!")
        else:
            self.alert(
                f"✔ Displaying {len(_words)} matching tokens. {'' if len(_words) < self.top_count else ' (result truncated)'}"
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
        self._picker.observe(self._plot_trends, names='value')

        self._compute.on_click(self._compute_keyness)

        self._smooth.observe(self._invalidate_handler, names='value')
        self._normalize.observe(self._invalidate_handler, names='value')
        self._keyness.observe(self._invalidate_handler, names='value')
        self._top_count.observe(self._invalidate_handler, names='value')
        self._time_period.observe(self._invalidate_handler, names='value')

        return self

    def buzy(self, value: bool) -> None:

        self._compute.disable = value
        self._smooth.disable = value
        self._normalize.disable = value
        self._words.disable = value
        self._keyness.disable = value
        self._top_count.disable = value
        self._time_period.disable = value

        # if value:
        #     with contextlib.suppress(Exception):
        #         self._words.unobserve(self._update_picker, names='value')
        #         self._tab.unobserve(self._plot_trends, 'selected_index')
        #         self._picker.unobserve(self._plot_trends, names='value')
        # else:
        #     self._words.observe(self._update_picker, names='value')
        #     self._tab.observe(self._plot_trends, 'selected_index')
        #     self._picker.observe(self._plot_trends, names='value')

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
        self._alert.value = msg

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
    def time_period(self) -> str:
        return self._time_period.value

    @property
    def top_count(self) -> int:
        return self._top_count.value

    @property
    def options(self) -> TrendsComputeOpts:
        return TrendsComputeOpts(
            normalize=self.normalize,
            smooth=self.smooth,
            keyness=self.keyness,
            time_period=self.time_period,
            top_count=self.top_count,
            words=self.words_or_regexp,
            descending=True,
        )


class TrendsGUI(TrendsBaseGUI):
    def keyness_widget(self) -> Dropdown:
        return Dropdown(
            options={
                "TF": KeynessMetric.TF,
                "TF (norm)": KeynessMetric.TF_normalized,
                "TF-IDF": KeynessMetric.TF_IDF,
            },
            value=KeynessMetric.TF,
            layout=Layout(width='auto'),
            tooltips=[
                "Show raw term frequency (TF) counts",
                "Show normalized word TF counts (normalize over time)",
                "Show TF-IDF weighed TF counts",
            ],
        )


class CoOccurrenceTrendsGUI(TrendsBaseGUI):
    def __init__(self, bundle: Bundle, n_top_count: int = 1000):
        super().__init__(n_top_count=n_top_count)
        self.bundle = bundle
        self._keyness_source: Dropdown = Dropdown(
            options={
                "Full corpus": KeynessMetricSource.Full,
                "Concept corpus": KeynessMetricSource.Concept,
                "Weighed corpus": KeynessMetricSource.Weighed,
            }
            if bundle.concept_corpus is not None
            else {
                "Full corpus": KeynessMetricSource.Full,
            },
            value=KeynessMetricSource.Weighed if bundle.concept_corpus is not None else KeynessMetricSource.Full,
            layout=Layout(width='auto'),
        )
        self._placeholder.children = [HTML("<b>Keyness source</b>"), self._keyness_source]

    def keyness_widget(self) -> Dropdown:
        return Dropdown(
            options={
                "TF": KeynessMetric.TF,
                "TF (norm)": KeynessMetric.TF_normalized,
                "TF-IDF": KeynessMetric.TF_IDF,
                "HAL CWR": KeynessMetric.HAL_cwr,
                "PPMI": KeynessMetric.PPMI,
                "LLR": KeynessMetric.LLR,
                "LLR(Z)": KeynessMetric.LLR_Z,
                "LLR(N)": KeynessMetric.LLR_N,
                "DICE": KeynessMetric.DICE,
            },
            value=KeynessMetric.TF,
            layout=Layout(width='auto'),
        )

    # def setup(self, *, displayers: Sequence[ITrendDisplayer]) -> "CoOccurrenceTrendsGUI":
    #     super().setup(displayers=displayers)
    #     self._keyness_source.observe(self._plot_trends, names='value')
    #     return self

    @property
    def keyness_source(self) -> KeynessMetricSource:
        return self._keyness_source.value

    @property
    def options(self) -> TrendsComputeOpts:
        trends_opts: TrendsComputeOpts = super().options
        trends_opts.keyness_source = self.keyness_source
        return trends_opts
