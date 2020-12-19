from dataclasses import dataclass, field
from typing import Sequence

import ipywidgets as widgets
from penelope.utility import get_logger

from .displayers import WORD_TREND_DISPLAYERS, ITrendDisplayer
from .trends_data import TrendsOpts, TrendsData

logger = get_logger()

BUTTON_LAYOUT = widgets.Layout(width='100px')
OUTPUT_LAYOUT = widgets.Layout(width='600px')


@dataclass
class TrendsGUI:
    """GUI component that displays word trends"""

    trends_data: TrendsData = field(default=None, init=False)

    _tab: widgets.Tab = widgets.Tab()
    _normalize: widgets.ToggleButton = widgets.ToggleButton(
        description="Normalize", icon='check', value=False, layout=BUTTON_LAYOUT
    )
    _smooth: widgets.ToggleButton = widgets.ToggleButton(
        description="Smooth", icon='check', value=False, layout=BUTTON_LAYOUT
    )
    _group_by: widgets.Dropdown = widgets.Dropdown(
        options=['year', 'lustrum', 'decade'],
        value='year',
        description='',
        disabled=False,
        layout=widgets.Layout(width='75px'),
    )
    _status: widgets.Label = widgets.Label(layout=widgets.Layout(width='50%', border="0px transparent white"))
    _words: widgets.Textarea = widgets.Textarea(
        description="",
        rows=2,
        value="",
        placeholder='Enter words and/or reg.exps. such as |.*ment$|',
        layout=widgets.Layout(width='98%'),
    )
    _word_count: widgets.BoundedIntText = widgets.BoundedIntText(
        value=10, min=3, max=100, step=1, description='Max words:', disabled=False
    )
    _displayers: Sequence[ITrendDisplayer] = field(default_factory=list)

    # update_handler: Callable = field(default=None, init=False)

    def layout(self) -> widgets.VBox:
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        self._normalize,
                        self._smooth,
                        self._group_by,
                        self._status,
                    ]
                ),
                self._words,
                self._tab,
            ]
        )

    def _plot_trends(self, *_):

        try:

            if self.trends_data is None or self.trends_data.corpus is None:
                self.alert("Please load a corpus!")
                return

            self.current_displayer.display(
                corpus=self.trends_data.get_corpus(self.normalize, self.group_by),
                indices=self.trends_data.find_indices(self.options),
                smooth=self.smooth,
            )

            self.alert("✔")

        except ValueError as ex:
            self.alert(str(ex))
        except Exception as ex:
            logger.exception(ex)
            self.warn(str(ex))

    def setup(self, *, displayers: Sequence[ITrendDisplayer] = None) -> "TrendsGUI":

        displayers = displayers or WORD_TREND_DISPLAYERS

        for i, cls in enumerate(displayers):
            displayer: ITrendDisplayer = cls()
            self._displayers.append(displayer)
            displayer.output = widgets.Output()
            with displayer.output:
                displayer.setup()

        self._tab.children = [d.output for d in self._displayers]
        for i, d in enumerate(self._displayers):
            self._tab.set_title(i, d.name)

        self._words.observe(self._plot_trends, names='value')
        self._tab.observe(self._plot_trends, 'selected_index')
        self._normalize.observe(self._plot_trends, names='value')
        self._smooth.observe(self._plot_trends, names='value')
        self._group_by.observe(self._plot_trends, names='value')

        return self

    def display(self, *, trends_data: TrendsData):
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
    def words(self):
        return ' '.join(self._words.value.split()).split()

    @property
    def smooth(self) -> bool:
        return self._smooth.value

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
            group_by=self.group_by,
            word_count=self.word_count,
            words=self.words,
        )
