import contextlib
from collections.abc import Iterable
from typing import List, Set, Union

import IPython.display as IPython_display
import pandas as pd
from ipywidgets import HTML, Button, Dropdown, GridBox, HBox, Layout, Output, Text, ToggleButton, VBox
from penelope.co_occurrence import Bundle, CoOccurrenceHelper, store_co_occurrences
from penelope.corpus import Token2Id
from penelope.corpus.dtm.vectorized_corpus import VectorizedCorpus
from penelope.notebook.utility import create_js_download
from penelope.utility import path_add_timestamp
from perspective import PerspectiveWidget


# pylint: disable=too-many-instance-attributes


def get_prepared_corpus(
    corpus: VectorizedCorpus,
    period_specifier: str,
    tf_idf: bool,
    token_filter: str,
    global_threshold: Union[int, float],
) -> VectorizedCorpus:

    corpus = corpus.group_by_document_index(period_specifier=period_specifier)

    if global_threshold > 1:
        corpus = corpus.slice_by_term_frequency(global_threshold)

    if tf_idf:
        corpus = corpus.tf_idf()

    if token_filter:
        indices = corpus.find_matching_words_indices(token_filter, n_max_count=None)
        corpus = corpus.slice_by_indicies(indices)

    return corpus


class CoOccurrenceTable(GridBox):  # pylint: disable=too-many-ancestors
    def __init__(
        self,
        *,
        bundle: Bundle,
        default_token_filter: str = None,
        **kwargs,
    ):
        """Alternative implementation that uses VectorizedCorpus"""
        self.bundle = bundle
        self.co_occurrences: pd.DataFrame = None

        if not isinstance(bundle.token2id, Token2Id):
            raise ValueError(f"Expected Token2Id, found {type(bundle.token2id)}")

        if not isinstance(bundle.compute_options, dict):
            raise ValueError("Expected Compute Options in bundle but found no such thing.")

        """Current processed corpus"""
        self.corpus = bundle.corpus

        """Properties that changes current corpus"""
        self._pivot: Dropdown = Dropdown(
            options=["year", "lustrum", "decade"],
            value="decade",
            placeholder='Group by',
            layout=Layout(width='auto'),
        )

        self._tf_idf = ToggleButton(description='TF_IDF', value=False, icon='', layout=Layout(width='auto'))

        """Properties that don't change current corpus"""
        # self._rank: Dropdown = Dropdown(
        #     options=[10 ** i for i in range(0, 7)],
        #     value=10000,
        #     layout=Layout(width='auto'),
        # )
        self._token_filter: Text = Text(
            value=default_token_filter, placeholder='token match', layout=Layout(width='auto')
        )
        self._global_threshold_filter: Dropdown = Dropdown(
            options={f'>= {i}': i for i in (1, 2, 3, 4, 5, 10, 25, 50, 100, 250, 500)},
            value=5,
            layout=Layout(width='auto'),
        )
        self.concepts: Set[str] = set(self.bundle.context_opts.concept or [])
        self._largest: Dropdown = Dropdown(
            options=[10 ** i for i in range(0, 7)],
            value=10000,
            layout=Layout(width='auto'),
        )
        self._show_concept = ToggleButton(description='Show concept', value=False, icon='', layout=Layout(width='auto'))
        self._message: HTML = HTML()
        self._save = Button(description='Save data', layout=Layout(width='auto'))
        self._display = Button(description='Update', layout=Layout(width='auto'))
        self._download = Button(description='Download data', layout=Layout(width='auto'))
        self._download_output: Output = Output()

        self._table: PerspectiveWidget = PerspectiveWidget(None)  # self.co_occurrences, sort=[["token", "asc"]], aggregates={}, )

        self._button_bar = HBox(
            children=[
                VBox([HTML("<b>Token match</b>"), self._token_filter]),
                VBox([self._tf_idf, self._show_concept if self._show_concept is not None else HTML("")]),
                VBox([HTML("<b>Group by</b>"), self._pivot]),
                VBox([HTML("<b>Global threshold</b>"), self._global_threshold_filter]),
                VBox([HTML("<b>Group limit</b>"), self._largest]),
                # VBox([HTML("<b>Group ranks</b>"), self._rank]),
                # VBox([HTML("<b>Result limit</b>"), self._head]),
                VBox([self._save, self._download]),
                VBox([self._display]),
                VBox([HTML("😢"), self._message]),
                self._download_output,
            ],
            layout=Layout(width='auto'),
        )
        super().__init__(children=[self._button_bar, self._table], layout=Layout(width='auto'), **kwargs)

        self._save.on_click(self.save)
        self._display.on_click(self.save)
        self._download.on_click(self.download)

        self.start_observe()

    def start_observe(self):

        self.stop_observe()

        self._tf_idf.observe(self._update_corpus, 'value')
        self._tf_idf.observe(self._update_toggle_icon, 'value')

        self._show_concept.observe(self._update_co_occurrences, 'value')
        self._show_concept.observe(self._update_toggle_icon, 'value')

        self._pivot.observe(self._update_corpus, 'value')

        self._global_threshold_filter.observe(self._update_co_occurrences, 'value')
        # self._head.observe(self._update_co_occurrences, 'value')
        self._token_filter.observe(self._update_co_occurrences, 'value')

        return self

    def stop_observe(self):

        with contextlib.suppress(Exception):

            self._tf_idf.unobserve(self._update_corpus, 'value')
            self._tf_idf.unobserve(self._update_toggle_icon, 'value')

            self._show_concept.unobserve(self._update_co_occurrences, 'value')
            self._show_concept.unobserve(self._update_toggle_icon, 'value')

            self._pivot.unobserve(self._update_corpus, 'value')

            self._global_threshold_filter.unobserve(self._update_co_occurrences, 'value')
            # self._head.unobserve(self._update_co_occurrences, 'value')
            self._token_filter.unobserve(self._update_co_occurrences, 'value')

    def alert(self, message: str) -> None:
        self._message.value = f"<span style='color: red; font-weight: bold;'>{message}</span>"

    def info(self, message: str) -> None:
        self._message.value = f"<span style='color: green; font-weight: bold;'>{message}</span>"

    def _update_corpus(self, *_):

        self.corpus = self.to_corpus()

        self._update_co_occurrences()

    def _update_co_occurrences(self, *_) -> pd.DataFrame:

        self.co_occurrences= self.to_co_occurrences()

        self.info(f"Data size: {len(self.co_occurrences)}")

        self._table.load(self.co_occurrences)

    def _update_toggle_icon(self, event: dict) -> None:
        with contextlib.suppress(Exception):
            event['owner'].icon = 'check' if event['new'] else ''

    @property
    def ignores(self) -> List[str]:

        if self.show_concept or not self.concepts:
            return set()

        if isinstance(self.concepts, Iterable):
            return set(self.concepts)

        return {self.concepts}

    @property
    def show_concept(self) -> bool:
        return self._show_concept.value

    @show_concept.setter
    def show_concept(self, value: bool):
        self._show_concept.value = value

    @property
    def tf_idf(self) -> bool:
        return self._tf_idf.value

    @tf_idf.setter
    def tf_idf(self, value: bool):
        self._tf_idf.value = value

    @property
    def global_threshold(self) -> int:
        return self._global_threshold_filter.value

    @global_threshold.setter
    def global_threshold(self, value: int):
        self._global_threshold_filter.value = value

    @property
    def largest(self) -> int:
        return self._largest.value

    @largest.setter
    def largest(self, value: int):
        self._largest.value = value

    @property
    def token_filter(self) -> List[str]:
        return self._token_filter.value.strip().split()

    @token_filter.setter
    def token_filter(self, value: str):
        self._token_filter.value = value

    @property
    def pivot(self) -> str:
        return self._pivot.value

    @pivot.setter
    def pivot(self, value: str):
        self._pivot.value = value

    def save(self, *_b):
        store_co_occurrences(
            filename=path_add_timestamp('co_occurrence_data.csv'),
            co_occurrences=self.co_occurrences,
            store_feather=False,
        )

    def download(self, *_):
        self._button_bar.disabled = True

        with contextlib.suppress(Exception):
            js_download = create_js_download(self.co_occurrences, index=True)
            if js_download is not None:
                with self._download_output:
                    IPython_display.display(js_download)

        self._button_bar.disabled = False


    def to_co_occurrences(self) -> pd.DataFrame:

        co_occurrences: pd.DataFrame = (
            CoOccurrenceHelper(
                self.corpus.to_co_occurrences(self.bundle.token2id),
                self.bundle.token2id,
                self.bundle.document_index,
                pivot_keys=self.pivot,
            )
            .groupby(self.pivot)
            .exclude(self.ignores)
            .largest(self.largest)
        ).value

        return co_occurrences


    def to_corpus(self) -> VectorizedCorpus:

        corpus: VectorizedCorpus = get_prepared_corpus(
            corpus=self.bundle.corpus,
            period_specifier=self.pivot,
            tf_idf=self.tf_idf,
            token_filter=self.token_filter,
            global_threshold=self.global_threshold,
        )
        return corpus
