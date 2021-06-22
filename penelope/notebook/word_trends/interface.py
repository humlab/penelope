import abc
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd
from penelope.co_occurrence import Bundle
from penelope.co_occurrence.keyness import ComputeKeynessOpts
from penelope.common.goodness_of_fit import GofData
from penelope.common.keyness import KeynessMetric, KeynessMetricSource
from penelope.corpus import VectorizedCorpus


@dataclass
class TrendsComputeOpts:

    normalize: bool
    keyness: KeynessMetric
    time_period: str

    fill_gaps: bool = False
    smooth: bool = None
    top_count: int = None
    words: List[str] = None
    descending: bool = False
    keyness_source: KeynessMetricSource = KeynessMetricSource.Full

    @property
    def clone(self) -> "TrendsComputeOpts":
        return TrendsComputeOpts(**asdict(self))

    def invalidates_corpus(self, other: "TrendsComputeOpts") -> bool:
        if self.normalize != other.normalize:
            return True
        if self.keyness != other.keyness:
            return True
        if self.time_period != other.time_period:
            return True
        if self.fill_gaps != other.fill_gaps:
            return True
        if self.keyness_source != other.keyness_source:
            return True
        return False


class ITrendsData(abc.ABC):
    def __init__(self, corpus: VectorizedCorpus, corpus_folder: str, corpus_tag: str, n_count: int = 100000):
        self.corpus: VectorizedCorpus = corpus
        self.corpus_folder: str = corpus_folder
        self.corpus_tag: str = corpus_tag
        self.n_count: int = n_count

        self._compute_options: Dict = None
        self._gof_data: GofData = None

        self._transformed_corpus: VectorizedCorpus = None
        self._trends_opts: TrendsComputeOpts = TrendsComputeOpts(
            normalize=False, keyness=KeynessMetric.TF, time_period='year'
        )
        self.category_column: str = "time_period"

    @abc.abstractmethod
    def _transform_corpus(self, opts: TrendsComputeOpts) -> VectorizedCorpus:
        ...

    @property
    def transformed_corpus(self) -> VectorizedCorpus:
        return self._transformed_corpus

    @property
    def gof_data(self) -> GofData:
        if self._gof_data is None:
            self._gof_data = GofData.compute(self.corpus, n_count=self.n_count)
        return self._gof_data

    def find_word_indices(self, opts: TrendsComputeOpts) -> List[int]:
        indices: List[int] = self._transform_corpus(opts).find_matching_words_indices(
            opts.words, opts.top_count, descending=opts.descending
        )
        return indices

    def find_words(self, opts: TrendsComputeOpts) -> List[str]:
        words: List[int] = self._transform_corpus(opts).find_matching_words(
            opts.words, opts.top_count, descending=opts.descending
        )
        return words

    def get_top_terms(self, n_count: int = 100, kind='token+count') -> pd.DataFrame:
        top_terms = self._transformed_corpus.get_top_terms(
            category_column=self.category_column, n_count=n_count, kind=kind
        )
        return top_terms

    def transform(self, opts: TrendsComputeOpts) -> "ITrendsData":

        if self._transformed_corpus is not None:
            if not self._trends_opts.invalidates_corpus(opts):
                return self

        self._transformed_corpus = self._transform_corpus(opts)
        self._trends_opts = opts.clone
        self._gof_data = None

        return self

    def reset(self) -> "ITrendsData":
        self._transformed_corpus = None
        self._trends_opts = TrendsComputeOpts(normalize=False, keyness=KeynessMetric.TF, time_period='year')
        self._gof_data = None
        return self


class TrendsData(ITrendsData):
    def __init__(self, corpus: VectorizedCorpus, corpus_folder: str, corpus_tag: str, n_count: int = 100000):
        super().__init__(corpus=corpus, corpus_folder=corpus_folder, corpus_tag=corpus_tag, n_count=n_count)

    def _transform_corpus(self, opts: TrendsComputeOpts) -> VectorizedCorpus:

        transformed_corpus: VectorizedCorpus = self.corpus

        """ Normal word trends """
        if opts.keyness == KeynessMetric.TF_IDF:
            transformed_corpus = transformed_corpus.tf_idf()
        elif opts.keyness == KeynessMetric.TF_normalized:
            transformed_corpus = transformed_corpus.normalize_by_raw_counts()

        transformed_corpus = transformed_corpus.group_by_time_period(
            time_period_specifier=opts.time_period,
            target_column_name=self.category_column,
            fill_gaps=opts.fill_gaps,
        )

        if opts.normalize:
            transformed_corpus = transformed_corpus.normalize_by_raw_counts()

        return transformed_corpus


class BundleTrendsData(ITrendsData):
    def __init__(self, bundle: Bundle = None, n_count: int = 100000):
        super().__init__(corpus=bundle.corpus, corpus_folder=bundle.folder, corpus_tag=bundle.tag, n_count=n_count)
        self.bundle = bundle
        self.keyness_source: KeynessMetricSource = KeynessMetricSource.Full
        self.tf_threshold: int = 1

    def _transform_corpus(self, opts: TrendsComputeOpts) -> VectorizedCorpus:

        transformed_corpus: VectorizedCorpus = self.bundle.keyness_transform(
            opts=ComputeKeynessOpts(
                period_pivot=opts.time_period,
                tf_threshold=self.tf_threshold,
                keyness_source=self.keyness_source,
                keyness=opts.keyness,
                pivot_column_name=self.category_column,
                normalize=opts.normalize,
                fill_gaps=opts.fill_gaps,
            )
        )

        return transformed_corpus
