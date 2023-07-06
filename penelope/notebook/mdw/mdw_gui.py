import logging
from typing import Callable

import pandas as pd
from ipywidgets import HTML, Button, HBox, IntRangeSlider, IntSlider, VBox

from penelope.corpus import dtm
from penelope.vendor import textacy_api as mdw

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes


class MDW_GUI:
    def __init__(self):
        self._top_n_terms: IntSlider = IntSlider(
            description='',
            min=10,
            max=1000,
            value=100,
            tooltip='The total number of most discriminating terms to return for each group',
            layout={'width': '250px'},
        )
        self._max_n_terms: IntSlider = IntSlider(
            description='',
            min=1,
            max=2000,
            value=2000,
            tooltip='Only consider terms whose document frequency is within the top # terms out of all terms',
            layout={'width': '250px'},
        )
        self._period1: IntRangeSlider = IntRangeSlider(
            description='',
            min=1900,
            max=2099,
            value=(2001, 2002),
            layout={'width': '250px'},
        )
        self._period2 = IntRangeSlider(
            description='',
            min=1900,
            max=2099,
            value=(2001, 2002),
            layout={'width': '250px'},
        )

        self._compute = Button(description='Compute', icon='', button_style='Success', layout={'width': '120px'})
        self.compute_callback: Callable = default_compute_callback
        self.done_callback: Callable = None
        self.corpus: dtm.VectorizedCorpus = None

    def setup(self, corpus: dtm.VectorizedCorpus, done_callback: Callable) -> "MDW_GUI":
        low, high = corpus.document_index.year.min(), corpus.document_index.year.max()

        self._period1.min, self._period1.max = (low, high)
        self._period2.min, self._period2.max = (low, high)
        self._period1.value = (low, low + 4)
        self._period2.value = (high - 4, high)

        self._compute.on_click(self._compute_handler)

        self.done_callback = done_callback
        self.corpus = corpus

        return self

    def layout(self):
        _layout = VBox(
            [
                HBox(
                    [
                        VBox(
                            [
                                HTML("<b>Period for first group</b>"),
                                self._period1,
                                HTML("<b>Period for second group</b>"),
                                self._period2,
                            ]
                        ),
                        VBox(
                            [
                                HTML("<b>Terms to display</b>"),
                                self._top_n_terms,
                                HTML("<b>Top TF threshold</b>"),
                                self._max_n_terms,
                            ],
                        ),
                        VBox([HTML("&nbsp;<p/>&nbsp;<p/>&nbsp;<p/>"), self._compute]),
                    ]
                ),
            ]
        )

        return _layout

    def _compute_handler(self, *_):
        if self.compute_callback is None:
            return

        try:
            self._compute.disabled = True

            df_mdw = self.compute_callback(self.corpus, self)

            if self.done_callback is not None:
                self.done_callback(self.corpus, df_mdw)

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


def default_compute_callback(corpus: dtm.VectorizedCorpus, args: MDW_GUI):
    group1_indices = corpus.document_index[corpus.document_index.year.between(*args.period1)].index
    group2_indices = corpus.document_index[corpus.document_index.year.between(*args.period2)].index

    df_dtm = mdw.compute_most_discriminating_terms(
        corpus,
        group1_indices=group1_indices,
        group2_indices=group2_indices,
        top_n_terms=args.top_n_terms,
        max_n_terms=args.max_n_terms,
    )

    return df_dtm
