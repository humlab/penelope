from dataclasses import dataclass
from typing import Dict

import pandas as pd
import penelope.common.goodness_of_fit as gof
from penelope.co_occurrence.convert import to_vectorized_corpus
from penelope.corpus import VectorizedCorpus

CO_OCCURRENCE_FILENAME_POSTFIX = '_co-occurrence.csv.zip'


@dataclass
class CoOccurrenceData:  # pylint: disable=too-many-instance-attributes

    corpus: VectorizedCorpus
    corpus_folder: str
    corpus_tag: str
    n_count: int

    co_occurrences: pd.DataFrame = None
    compute_options: Dict = None

    co_occurrences_metadata: Dict = None
    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None

    def update(
        self,
        co_occurrences: pd.DataFrame,
        corpus: VectorizedCorpus,
        corpus_folder: str,
        corpus_tag: str,
        n_count: int,
        compute_options: Dict,
    ) -> "CoOccurrenceData":

        self.corpus = corpus
        self.corpus_folder = corpus_folder
        self.co_occurrences = co_occurrences
        self.corpus_tag = corpus_tag
        self.compute_options = compute_options

        if self.corpus is None and self.co_occurrences is None:
            raise ValueError("Both corpus and co_occurrences cannot be None")

        if self.corpus is None:

            self.corpus = to_vectorized_corpus(
                co_occurrences=self.co_occurrences, value_column='value_n_t'
            ).group_by_year()

        self.goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(self.corpus, None, verbose=False)
        self.most_deviating_overview = gof.compile_most_deviating_words(self.goodness_of_fit, n_count=n_count)
        self.most_deviating = gof.get_most_deviating_words(
            self.goodness_of_fit, 'l2_norm', n_count=n_count, ascending=False, abs_value=True
        )

        return self
