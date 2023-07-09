from typing import Any, Protocol

import ipywidgets as w
import pandas as pd
from loguru import logger

from penelope.corpus import dtm
from penelope.vendor import textacy_api as mdw

# pylint: disable=too-many-instance-attributes


class ComputedCallback(Protocol):
    def __call__(self, *, result: dtm.VectorizedCorpus, sender: Any, **kwargs) -> None:
        ...


class ComputeCallback(Protocol):
    def __call__(self, *, corpus: dtm.VectorizedCorpus, args: "MDW_GUI") -> None:
        ...


class MDW_GUI:
    """GUI for computing MDW"""

    def __init__(self):
        self._top_n_terms: w.IntSlider = w.IntSlider(
            description='',
            min=10,
            max=1000,
            value=100,
            tooltip='The total number of most discriminating terms to return for each group',
            layout={'width': '250px'},
        )
        self._max_n_terms: w.IntSlider = w.IntSlider(
            description='',
            min=1,
            max=2000,
            value=2000,
            tooltip='Only consider terms whose document frequency is within the top # terms out of all terms',
            layout={'width': '250px'},
        )
        self._period1: w.IntRangeSlider = w.IntRangeSlider(
            description='',
            min=1900,
            max=2099,
            value=(2001, 2002),
            layout={'width': '250px'},
        )
        self._period2 = w.IntRangeSlider(
            description='',
            min=1900,
            max=2099,
            value=(2001, 2002),
            layout={'width': '250px'},
        )
        self._extra_placeholder: w.HBox = w.HBox([])

        self._compute = w.Button(
            description='Compute', icon='', button_style='Success', layout={'width': '120px'}, disabled=True
        )
        self.compute_callback: ComputeCallback = default_compute_mdw
        self.computed_callback: ComputedCallback = None

        self._corpus: dtm.VectorizedCorpus = None
        self.disabled = True

    def setup(self, computed_callback: ComputedCallback) -> "MDW_GUI":
        self._compute.on_click(self._compute_handler)
        self.computed_callback = computed_callback
        return self

    @property
    def corpus(self) -> dtm.VectorizedCorpus:
        return self._corpus

    @corpus.setter
    def corpus(self, value: dtm.VectorizedCorpus):
        low, high = value.document_index.year.min(), value.document_index.year.max()

        self._period1.min, self._period1.max = (low, high)
        self._period2.min, self._period2.max = (low, high)
        self._period1.value = (low, low + 4)
        self._period2.value = (high - 4, high)

        self._corpus = value

        self.disabled = False

        return self

    def layout(self):
        _layout = w.VBox(
            [
                w.HBox(
                    [
                        w.VBox(
                            [
                                w.HTML("<b>Period for first group</b>"),
                                self._period1,
                                w.HTML("<b>Period for second group</b>"),
                                self._period2,
                            ]
                        ),
                        w.VBox(
                            [
                                w.HTML("<b>Terms to display</b>"),
                                self._top_n_terms,
                                w.HTML("<b>Top TF threshold</b>"),
                                self._max_n_terms,
                            ],
                        ),
                        w.VBox([w.HTML("&nbsp;<p/>&nbsp;<p/>&nbsp;<p/>"), self._compute]),
                    ]
                ),
            ]
        )

        return _layout

    def add(self, widget: w.CoreWidget, append: bool = False):
        self._extra_placeholder.children = (list(self._extra_placeholder.children) if append else []) + [widget]

    def _compute_handler(self, *_):
        if self.compute_callback is None:
            return

        try:
            self._compute.disabled = True

            data: pd.DataFrame = self.compute_callback(self.corpus, self)

            if self.computed_callback is not None:
                self.computed_callback(result=data, sender=self)

        except Exception as ex:
            logger.error(ex)
        finally:
            self._compute.disabled = False

    @property
    def period1(self):
        return self._period1.value

    @property
    def period2(self):
        return self._period2.value

    @property
    def top_n_terms(self):
        return self._top_n_terms.value

    @property
    def max_n_terms(self):
        return self._max_n_terms.value

    @property
    def disabled(self) -> bool:
        return self._compute.disabled

    @disabled.setter
    def disabled(self, value: bool):
        self._compute.disabled = value
        self._period1.disabled = value
        self._period2.disabled = value
        self._top_n_terms.disabled = value
        self._max_n_terms.disabled = value


def default_compute_mdw(corpus: dtm.VectorizedCorpus, args: MDW_GUI) -> pd.DataFrame:
    """Computes most discriminating terms for two periods of time"""
    group1_indices = corpus.document_index[corpus.document_index.year.between(*args.period1)].index
    group2_indices = corpus.document_index[corpus.document_index.year.between(*args.period2)].index

    result: pd.DataFrame = mdw.compute_most_discriminating_terms(
        corpus,
        group1_indices=group1_indices,
        group2_indices=group2_indices,
        top_n_terms=args.top_n_terms,
        max_n_terms=args.max_n_terms,
    )

    return result
