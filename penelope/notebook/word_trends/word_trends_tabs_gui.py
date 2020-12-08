from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import ipywidgets as widgets
from penelope.common.curve_fit import pchip_spline  # , rolling_average_smoother
from penelope.utility import get_logger

from .displayers import WORD_TREND_DISPLAYERS, ITrendDisplayer, WordTrendData
from .utils import find_candidate_words, find_n_top_words

logger = get_logger()

DEFAULT_SMOOTHERS = [pchip_spline]  # , rolling_average_smoother('nearest', 3)]
BUTTON_LAYOUT = widgets.Layout(width='100px')
OUTPUT_LAYOUT = widgets.Layout(width='600px')


@dataclass
class TabsGUI:
    """Container for GUO components"""

    tab: widgets.Tab = widgets.Tab()
    normalize = widgets.ToggleButton(description="Normalize", icon='check', value=False, layout=BUTTON_LAYOUT)
    smooth = widgets.ToggleButton(description="Smooth", icon='check', value=False, layout=BUTTON_LAYOUT)
    status = widgets.Label(layout=widgets.Layout(width='50%', border="0px transparent white"))
    words = widgets.Textarea(
        description="",
        rows=2,
        value="",
        layout=widgets.Layout(width='90%'),
    )
    displayers: Sequence[ITrendDisplayer] = field(default_factory=list)

    trend_data: WordTrendData = field(default=None, init=False)
    update_handler: Callable = field(default=None, init=False)

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        self.normalize,
                        self.smooth,
                        self.status,
                    ]
                ),
                self.words,
                self.tab,
            ]
        )

    def _update_handler(self, *_):
        if self.update_handler is not None:
            self.update_handler(self, self.trend_data)

    def _toggle_normalize(self, *_):

        self._update_handler(_)

    def setup(self, trend_data: WordTrendData, displayers: Sequence[ITrendDisplayer], update_handler: Callable):

        self.trend_data = trend_data

        for i, cls in enumerate(displayers):
            displayer: ITrendDisplayer = cls(data=self.trend_data)
            self.displayers.append(displayer)
            displayer.output = widgets.Output()
            with displayer.output:
                displayer.setup()

        self.tab.children = [d.output for d in self.displayers]
        for i, d in enumerate(self.displayers):
            self.tab.set_title(i, d.name)

        self.update_handler = update_handler
        self.words.observe(self._update_handler, names='value')
        self.tab.observe(self._update_handler, 'selected_index')
        self.normalize.observe(self._toggle_normalize, names='value')
        self.smooth.observe(self._update_handler, names='value')

        return self

    def update(self):
        self._update_handler(None)

    @property
    def current_displayer(self):
        return self.displayers[self.tab.selected_index]

    @property
    def current_output(self):
        return self.current_displayer.output

    @property
    def current_words(self):
        return ' '.join(self.words.value.split()).split()


def update_plot(gui: TabsGUI, trend_data: WordTrendData):

    if trend_data.corpus is None:
        gui.status.value = "Please load a corpus!"
        return

    if gui.normalize.value:
        if trend_data.normalized_corpus is None:
            trend_data.normalized_corpus = trend_data.corpus.normalize()

    corpus = trend_data.normalized_corpus if gui.normalize.value else trend_data.corpus

    # pattern = re.compile("^.*tion$")
    # corpus: VectorizedCorpus = sample_corpus()
    # sliced_corpus = corpus.slice_by(px=pattern.match)

    words: List[str] = find_candidate_words(gui.current_words, corpus.token2id)

    if len(words) > 10:
        words = find_n_top_words(corpus.word_counts, words, 15)

    # Gather indicies for specified words
    indices = [corpus.token2id[token] for token in words if token in corpus.token2id]

    # # Give warning if any words not found
    # missing_tokens = [token for token in tokens if token not in corpus.token2id]

    # if len(missing_tokens) > 0:
    #     gui.status.value = f"Not found: {' '.join(missing_tokens)}"
    #     return

    if len(indices) == 0:
        gui.status.value = "Nothing to plot!"
        return

    gui.current_displayer.clear()
    with gui.current_output:
        data = gui.current_displayer.compile(corpus, indices, smoothers=DEFAULT_SMOOTHERS if gui.smooth.value else [])
        _ = gui.current_displayer.plot(data, state=trend_data)


def create_tabs_gui(trend_data: WordTrendData) -> TabsGUI:

    gui = TabsGUI().setup(trend_data=trend_data, displayers=WORD_TREND_DISPLAYERS, update_handler=update_plot)

    return gui
