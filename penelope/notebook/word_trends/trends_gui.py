import abc
from typing import List, Sequence

import ipywidgets as w
import pandas as pd
from loguru import logger

from penelope import utility as pu
from penelope.co_occurrence.bundle import Bundle
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.notebook import mixins as mx
from penelope.notebook import widgets_utils as wu

from .displayers import ITrendDisplayer
from .interface import TrendsComputeOpts, TrendsData

BUTTON_LAYOUT = w.Layout(width='120px')
OUTPUT_LAYOUT = w.Layout(width='600px')

# pylint: disable=too-many-instance-attributes,too-many-public-methods


def create_data_frame(plot_data: dict, category_name: str) -> pd.DataFrame:
    df = pd.DataFrame(data=plot_data)
    if category_name in df.columns:
        df = df[[category_name] + [x for x in df.columns if x != category_name]]
    return df


class TrendsBaseGUI(abc.ABC):
    """GUI component that displays word trends"""

    @abc.abstractmethod
    def keyness_widget(self) -> w.Dropdown:
        ...

    def __init__(self, n_top_count: int = 1000):
        super().__init__()
        self.trends_data: TrendsData = None
        self.display_opts: dict = dict(width=1000, height=600)
        self._tab: w.Tab = w.Tab(layout={'width': '80%'})
        self._picker: w.SelectMultiple = w.SelectMultiple(
            description="", options=[], value=[], rows=20, layout={'width': '180px'}
        )
        self._normalize: w.ToggleButton = w.ToggleButton(
            description="Normalize",
            icon='check',
            value=False,
            layout=BUTTON_LAYOUT,
            tooltip="Normalize entire corpus by selected period",
        )

        self._keyness: w.ToggleButton = self.keyness_widget()
        self._placeholder: w.VBox = w.VBox()
        self._widgets_placeholder: w.VBox = w.VBox()
        self._header_placeholder: w.HBox = w.HBox()
        self._sidebar_placeholder: w.VBox = w.VBox(children=[])
        self._sidebar_ctrls: List[w.CoreWidget] = [w.VBox([w.HTML("<b>Matched words</b>"), self._picker])]
        self._smooth: w.ToggleButton = w.ToggleButton(
            description="Smooth", icon='check', value=False, layout=BUTTON_LAYOUT
        )
        self._auto_compute: w.ToggleButton = w.ToggleButton(
            description="auto", icon='check', value=True, layout=BUTTON_LAYOUT
        )
        self._temporal_key: w.Dropdown = w.Dropdown(
            options=['year', 'lustrum', 'decade'],
            value='decade',
            description='',
            disabled=False,
            layout=w.Layout(width='75px'),
        )
        self._alert: w.HTML = w.HTML()
        self._words: w.Textarea = w.Textarea(
            description="",
            rows=1,
            value="",
            placeholder='Enter words, wildcards and/or regexps such as "information", "info*", "*ment",  "|.*tion$|"',
            layout=w.Layout(width='740px'),
            continuous_update=False,
        )
        self._top_count: w.BoundedIntText = w.BoundedIntText(
            value=n_top_count,
            min=3,
            max=100000,
            step=10,
            description='',
            disabled=False,
            layout={'width': '180px'},
        )
        self._compute: w.Button = w.Button(
            description="Compute", button_style='success', disabled=True, layout=BUTTON_LAYOUT
        )
        self._displayers: Sequence[ITrendDisplayer] = []

    def setup(self, *, displayers: Sequence[ITrendDisplayer]) -> "TrendsGUI":
        for i, cls in enumerate(displayers):
            displayer: ITrendDisplayer = cls(**self.display_opts)
            self._displayers.append(displayer)
            displayer.output = w.Output()
            with displayer.output:
                displayer.setup()

        self._tab.children = [d.output for d in self._displayers]
        for i, d in enumerate(self._displayers):
            self._tab.set_title(i, d.name)

        self._compute.on_click(self._transform_handler)

        self.observe(True)

        return self

    def transform(self):
        try:

            if self.trends_data is None:
                self.alert("😮 Please load a corpus (no trends data) !")
                return

            if self.trends_data is None or self.trends_data.corpus is None:
                self.alert("😥 Please load a corpus (no corpus in trends data) !")
                return

            self.buzy(True)

            self.alert("⌛ Computing...")
            self.trends_data.transform(self.options)
            self.alert("✔")

            self.invalidate(False)

            self.plot()
            self.alert("✔️")

        except ValueError as ex:
            self.alert(str(ex))
        except Exception as ex:
            logger.exception(ex)
            self.warn(str(ex))
            raise
        finally:
            self.buzy(False)

    def display(self, *, trends_data: TrendsData):
        # OMG WTF!???
        # if self._picker is not None:
        #     self._picker.values = []
        #     self._picker.options = []
        self.plot(trends_data=trends_data)

    def extract(self) -> Sequence[pd.DataFrame]:  # pylint: disable=unused-argument
        self.alert("⌛ Extracting data...")
        data: pd.DataFrame = self.trends_data.extract(indices=self.picked_indices)
        self.alert("")
        return [data]

    def plot(self, trends_data: TrendsData = None):
        try:

            if trends_data is not None:
                self.trends_data = trends_data

            if self.trends_data is None or self.trends_data.transformed_corpus is None:
                self.alert("🥱 (not computed)")
                return

            if len(self.picked_words) == 0:
                self.alert("🙃 Please specify tokens to plot")
                return

            data: List[pd.DataFrame] = self.extract()

            self.alert("⌛ Plotting...")
            self.plot_current_displayer(data)

            self.alert("🙂")

        except ValueError as ex:
            self.alert(f"😡 {str(ex)}")
            raise
        except Exception as ex:
            logger.exception(ex)
            self.warn(f"😡 {str(ex)}")
            raise

    def plot_current_displayer(self, data: List[pd.DataFrame] = None):
        self.current_displayer.clear()
        with self.current_displayer.output:
            self.current_displayer.plot(data=data, temporal_key=self.temporal_key)

    def layout(self) -> w.VBox:
        self._sidebar_placeholder.children = self._sidebar_ctrls
        layout: w.VBox = w.VBox(
            [
                self._header_placeholder,
                w.HBox(
                    [
                        w.VBox([w.HTML("<b>Keyness</b>"), self._keyness]),
                        self._placeholder,
                        w.VBox([w.HTML("<b>Top count</b>"), self._top_count]),
                        w.VBox([w.HTML("<b>Grouping</b>"), self._temporal_key]),
                        self._widgets_placeholder,
                        w.VBox([self._normalize, self._smooth]),
                        w.VBox([self._auto_compute, self._compute]),
                        w.VBox([w.HTML("📌"), self._alert]),
                    ],
                    layout={'width': '98%'},
                ),
                w.HBox(
                    [
                        w.VBox([w.HTML("<b>Words to find</b> (press <b>tab</b> to start search)"), self._words]),
                    ],
                    layout={'width': '99%'},
                ),
                w.HBox([self._sidebar_placeholder, self._tab], layout={'width': '100%'}),
            ],
            layout=w.Layout(width='auto'),
        )

        return layout

    def invalidate(self, value: bool = True):
        self._compute.disabled = not value
        # self._words.disabled = value
        if value:
            for displayer in self._displayers:
                displayer.clear()
            if self.auto_compute:
                self._transform_handler()
            else:
                self.alert(" 🗑 Data invalidated (press compute to update).")

    def buzy(self, value: bool) -> None:
        self._compute.disabled = value
        self._smooth.disabled = value
        self._normalize.disabled = value
        self._words.disabled = value
        self._keyness.disabled = value

    def observe(self, value: bool, **kwargs):  # pylint: disable=unused-argument
        """Register or unregisters widget event handlers"""
        if hasattr(super(), "observe"):
            getattr(super(), "observe")(value=value, **kwargs)
        wu.register_observer(self._words, handler=self._update_picker_handler, value=value, names='value')
        wu.register_observer(self._tab, handler=self._plot_handler, value=value, names='selected_index')
        wu.register_observer(self._picker, handler=self._plot_handler, value=value, names='value')
        wu.register_observer(self._auto_compute, handler=self._auto_compute_handler, value=value, names='value')
        for ctrl in [self._smooth, self._normalize, self._keyness, self._top_count, self._temporal_key]:
            wu.register_observer(ctrl, handler=self._invalidate_handler, value=value)

    def _transform_handler(self, *_):
        self.transform()

    def _plot_handler(self, *_):
        self.plot()

    def _display_handler(self, *_):
        self.plot()

    def _update_picker_handler(self, *_):

        self.observe(False)
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
        self.observe(True)

    def _auto_compute_handler(self, *_):
        self._compute.disabled = self.auto_compute
        self._auto_compute.icon = 'check' if self.auto_compute else ''
        if self.auto_compute:
            self._transform_handler()

    def _invalidate_handler(self, *_):
        self.invalidate(True)

    @property
    def current_displayer(self) -> ITrendDisplayer:
        return self._displayers[self._tab.selected_index]

    @property
    def current_output(self):
        return self.current_displayer.output

    @property
    def auto_compute(self) -> bool:
        return self._auto_compute.value

    def alert(self, msg: str):
        self._alert.value = msg

    def warn(self, msg: str):
        self.alert(f"<span style='color=red'>{msg}</span>")

    @property
    def words_or_regexp(self):
        return ' '.join(self._words.value.split()).split()

    @property
    def picked_words(self) -> Sequence[int]:
        return self._picker.value

    @property
    def picked_indices(self) -> Sequence[int]:
        if self.trends_data is None or self.trends_data.transformed_corpus is None:
            return []
        indices: List[int] = self.trends_data.transformed_corpus.token_indices(self.picked_words)
        return indices

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
    def temporal_key(self) -> str:
        return self._temporal_key.value

    @property
    def top_count(self) -> int:
        return self._top_count.value

    @property
    def options(self) -> TrendsComputeOpts:
        return TrendsComputeOpts(
            normalize=self.normalize,
            smooth=self.smooth,
            keyness=self.keyness,
            temporal_key=self.temporal_key,
            top_count=self.top_count,
            words=self.words_or_regexp,
            descending=True,
        )


class TrendsGUI(mx.PivotKeysMixIn, TrendsBaseGUI):
    def __init__(
        self, pivot_key_specs: mx.PivotKeySpecArg = None, n_top_count: int = 1000
    ):  # pylint: disable=useless-super-delegation
        super().__init__(pivot_key_specs=pivot_key_specs, n_top_count=n_top_count)

    def extract(self) -> Sequence[pd.DataFrame]:
        self.alert("⌛ Extracting data...")
        """Fetch trends data grouped by pivot keys (ie. stacked)"""
        stacked_data: pd.DataFrame = self.trends_data.extract(indices=self.picked_indices, filter_opts=self.filter_opts)
        """Decode integer/coded pivot keys"""
        stacked_data = self.pivot_keys.decode_pivot_keys(stacked_data, True)

        pivot_keys: List[str] = [self.temporal_key] + [
            x for x in stacked_data.columns if x in self.pivot_keys.text_names
        ]
        if len(pivot_keys) > 1:
            """Unstack pivot keys (make columnar"""
            unstacked_data: pd.DataFrame = pu.unstack_data(stacked_data, pivot_keys=pivot_keys)
            return [stacked_data, unstacked_data]
        return [stacked_data]

    def keyness_widget(self) -> w.Dropdown:
        return w.Dropdown(
            options={
                "TF": KeynessMetric.TF,
                "TF (norm)": KeynessMetric.TF_normalized,
                "TF-IDF": KeynessMetric.TF_IDF,
            },
            value=KeynessMetric.TF,
            layout=w.Layout(width='auto'),
            tooltips=[
                "Show raw term frequency (TF) counts",
                "Show normalized word TF counts (normalize over time)",
                "Show TF-IDF weighed TF counts",
            ],
        )

    @property
    def options(self) -> TrendsComputeOpts:
        opts: TrendsComputeOpts = super().options
        opts.update(
            pivot_keys_id_names=self.pivot_keys_id_names,
            filter_opts=self.filter_opts,
            unstack_tabular=self.unstack_tabular,
        )
        return opts

    def buzy(self, value: bool) -> None:
        super().buzy(value)
        self._unstack_tabular.disabled = value
        self._multi_pivot_keys_picker.disabled = value
        self._filter_keys.disabled = value

    def observe(self, value: bool, **kwargs) -> None:  # pylint: disable=arguments-differ
        super().observe(value=value, handler=self._display_handler, **kwargs)
        for ctrl in [self._multi_pivot_keys_picker, self._filter_keys]:
            wu.register_observer(ctrl, handler=self._invalidate_handler, value=value)


class CoOccurrenceTrendsGUI(TrendsBaseGUI):
    def __init__(self, bundle: Bundle, n_top_count: int = 1000):
        super().__init__(n_top_count=n_top_count)
        self.bundle = bundle
        self._keyness_source: w.Dropdown = w.Dropdown(
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
            layout=w.Layout(width='auto'),
        )
        self._placeholder.children = [w.HTML("<b>Keyness source</b>"), self._keyness_source]

    def keyness_widget(self) -> w.Dropdown:
        return w.Dropdown(
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
            layout=w.Layout(width='auto'),
        )

    # def setup(self, *, displayers: Sequence[ITrendDisplayer]) -> "CoOccurrenceTrendsGUI":
    #     super().setup(displayers=displayers)
    #     self._keyness_source.observe(self._plot, names='value')
    #     return self

    @property
    def keyness_source(self) -> KeynessMetricSource:
        return self._keyness_source.value

    @property
    def options(self) -> TrendsComputeOpts:
        trends_opts: TrendsComputeOpts = super().options
        trends_opts.keyness_source = self.keyness_source
        return trends_opts
