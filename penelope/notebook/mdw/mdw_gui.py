import logging
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from ipywidgets import Button, HBox, IntRangeSlider, IntSlider, Layout, VBox
from penelope.corpus import dtm
from penelope.vendor.textacy import mdw_modified

logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes


@dataclass
class MDW_GUI:

    _top_n_terms: IntSlider = IntSlider(
        description='#terms',
        min=10,
        max=1000,
        value=100,
        tooltip='The total number of most discriminating terms to return for each group',
    )
    _max_n_terms: IntSlider = IntSlider(
        description='#top',
        min=1,
        max=2000,
        value=2000,
        tooltip='Only consider terms whose document frequency is within the top # terms out of all terms',
    )
    _period1: IntRangeSlider = IntRangeSlider(
        description='Period',
        min=1900,
        max=2099,
        value=(2001, 2002),
        layout={'width': '250px'},
    )
    _period2 = IntRangeSlider(
        description='Period',
        min=1900,
        max=2099,
        value=(2001, 2002),
        layout={'width': '250px'},
    )

    _compute = Button(description='Compute', icon='', button_style='Success', layout={'width': '120px'})

    compute_callback: Callable = None
    done_callback: Callable = None
    corpus: dtm.VectorizedCorpus = None

    def setup(self, corpus: dtm.VectorizedCorpus, compute_callback: Callable, done_callback: Callable) -> "MDW_GUI":

        low, high = corpus.document_index.year.min(), corpus.document_index.year.max()

        self._period1.min, self._period1.max = (low, high)
        self._period2.min, self._period2.max = (low, high)
        self._period1.value = (low, low + 4)
        self._period2.value = (high - 4, high)

        self._compute.on_click(self._compute_handler)

        self.compute_callback = compute_callback
        self.done_callback = done_callback
        self.corpus = corpus

        return self

    def layout(self):

        _layout = VBox(
            [
                HBox(
                    [
                        VBox([self._period1, self._period2]),
                        VBox(
                            [
                                self._top_n_terms,
                                self._max_n_terms,
                            ],
                            layout=Layout(align_items='flex-end'),
                        ),
                        VBox([self._compute]),
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

    df_dtm = mdw_modified.compute_most_discriminating_terms(
        corpus,
        group1_indices=group1_indices,
        group2_indices=group2_indices,
        top_n_terms=args.top_n_terms,
        max_n_terms=args.max_n_terms,
    )

    return df_dtm


def create_mdw_gui(
    corpus: dtm.VectorizedCorpus,
    done_callback: Callable[[dtm.VectorizedCorpus, pd.DataFrame], None],
    compute_callback: Callable[[dtm.VectorizedCorpus, pd.DataFrame], None] = None,
) -> MDW_GUI:

    gui = MDW_GUI().setup(
        corpus=corpus,
        compute_callback=(compute_callback or default_compute_callback),
        done_callback=done_callback,
    )

    return gui
