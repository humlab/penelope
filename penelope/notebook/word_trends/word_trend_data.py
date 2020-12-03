from dataclasses import dataclass
from typing import Dict

import pandas as pd
import penelope.common.goodness_of_fit as gof
from penelope.corpus import VectorizedCorpus


@dataclass
class WordTrendData:
    corpus: VectorizedCorpus = None
    corpus_folder: str = None
    corpus_tag: str = None
    compute_options: Dict = None
    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None

    def update(
        self,
        *,
        corpus: VectorizedCorpus = None,
        corpus_folder: str = None,
        corpus_tag: str = None,
        n_count: int = 25000,
    ) -> "WordTrendData":
        self.corpus = corpus
        self.corpus_folder = corpus_folder
        self.corpus_tag = corpus_tag
        self.compute_options = VectorizedCorpus.load_options(tag=corpus_tag, folder=corpus_folder)
        self.goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(
            self.corpus, None, verbose=True, metrics=['l2_norm', 'slope']
        )
        self.most_deviating_overview = gof.compile_most_deviating_words(self.goodness_of_fit, n_count=n_count)
        self.most_deviating = gof.get_most_deviating_words(
            self.goodness_of_fit, 'l2_norm', n_count=n_count, ascending=False, abs_value=True
        )
        return self
