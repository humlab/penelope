from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd
import penelope.common.goodness_of_fit as gof
from penelope.co_occurrence import Bundle
from penelope.common.keyness import KeynessMetric
from penelope.corpus import VectorizedCorpus


@dataclass
class TrendsOpts:

    normalize: bool
    keyness: KeynessMetric
    group_by: str

    smooth: bool = None
    word_count: int = None
    words: List[str] = None
    descending: bool = False

    @property
    def clone(self) -> "TrendsOpts":
        return TrendsOpts(**asdict(self))

    def invalidates_corpus(self, other: "TrendsOpts") -> bool:
        if self.normalize != other.normalize:
            return True
        if self.keyness != other.keyness:
            return True
        if self.group_by != other.group_by:
            return True
        return False


@dataclass
class GoodnessOfFitData:

    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None

    @staticmethod
    def compute(corpus: VectorizedCorpus, n_count: int) -> "GoodnessOfFitData":

        goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(
            corpus, None, verbose=True, metrics=['l2_norm', 'slope']
        )
        most_deviating_overview = gof.compile_most_deviating_words(goodness_of_fit, n_count=n_count)
        most_deviating = gof.get_most_deviating_words(
            goodness_of_fit, 'l2_norm', n_count=n_count, ascending=False, abs_value=True
        )

        gof_data: GoodnessOfFitData = GoodnessOfFitData(
            goodness_of_fit=goodness_of_fit,
            most_deviating=most_deviating,
            most_deviating_overview=most_deviating_overview,
        )

        return gof_data


class TrendsData:
    """Container class for displayed token trend

    Note: If `Bundle` is set (not None) then the trends data is co-occurrence data

    """

    def __init__(
        self,
        bundle: Bundle = None,
        corpus: VectorizedCorpus = None,
        corpus_folder: str = None,
        corpus_tag: str = None,
        n_count: int = 100000,
    ):
        if (corpus is None) == (bundle is None):
            raise ValueError("Bundle and Corpus are mutuala exclusive arguments!")

        self.bundle: Bundle = bundle

        self.corpus: VectorizedCorpus = corpus or bundle.corpus
        self.corpus_folder: str = corpus_folder or bundle.folder
        self.corpus_tag: str = corpus_tag or bundle.tag

        self.n_count: int = n_count

        self._compute_options: Dict = None
        self._gof_data: GoodnessOfFitData = None

        self.transformed_corpus: VectorizedCorpus = self.corpus
        self.current_trends_opts: TrendsOpts = TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='year')
        self.category_column: str = "time_period"

        # FIXME:
        # self.corpus = self.corpus.group_by_year(target_column_name=self.category_column)

    @property
    def gof_data(self) -> GoodnessOfFitData:
        if self._gof_data is None:
            self._gof_data = GoodnessOfFitData.compute(self.corpus, n_count=self.n_count)
        return self._gof_data

    def get_corpus(self, opts: TrendsOpts) -> VectorizedCorpus:

        if self.current_trends_opts.invalidates_corpus(opts):

            self._gof_data = None

            transformed_corpus: VectorizedCorpus = self.corpus

            if self.bundle is not None:
                """ Co-occurrence trends word trends """
                transformed_corpus = self.bundle.to_keyness_corpus(
                    period_pivot=opts.group_by,
                    global_threshold=1,
                    keyness=opts.keyness,
                    pivot_column_name=self.category_column,
                    normalize=opts.normalize,
                )
            else:
                """ Normal word trends """
                if opts.keyness == KeynessMetric.TF_IDF:
                    transformed_corpus = transformed_corpus.tf_idf()
                elif opts.keyness == KeynessMetric.TF_normalized:
                    transformed_corpus = transformed_corpus.normalize_by_raw_counts()

                transformed_corpus = transformed_corpus.group_by_time_period(
                    time_period_specifier=opts.group_by,
                    target_column_name=self.category_column,
                )

                if opts.normalize:
                    transformed_corpus = transformed_corpus.normalize_by_raw_counts()

            self.transformed_corpus = transformed_corpus
            self.current_trends_opts = opts.clone

        return self.transformed_corpus

    def find_word_indices(self, opts: TrendsOpts) -> List[int]:
        indices: List[int] = self.get_corpus(opts).find_matching_words_indices(
            opts.words, opts.word_count, descending=opts.descending
        )
        return indices

    def find_words(self, opts: TrendsOpts) -> List[str]:
        words: List[int] = self.get_corpus(opts).find_matching_words(
            opts.words, opts.word_count, descending=opts.descending
        )
        return words

    def get_top_terms(self, n_count: int = 100, kind='token+count') -> pd.DataFrame:
        top_terms = self.transformed_corpus.get_top_terms(
            category_column=self.category_column, n_count=n_count, kind=kind
        )
        return top_terms
