import abc
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
from penelope import corpus as pc
from penelope import utility as pu
from penelope.co_occurrence import Bundle
from penelope.co_occurrence.keyness import ComputeKeynessOpts
from penelope.common import goodness_of_fit as gof
from penelope.common import keyness as pk


@dataclass
class TrendsComputeOpts:

    normalize: bool
    keyness: pk.KeynessMetric

    temporal_key: str
    pivot_keys_id_names: List[str] = field(default_factory=list)
    pivot_keys_filter: pu.PropertyValueMaskingOpts = None
    unstack_tabular: bool = False

    fill_gaps: bool = False
    smooth: bool = None
    top_count: int = None
    words: List[str] = None
    descending: bool = False
    keyness_source: pk.KeynessMetricSource = pk.KeynessMetricSource.Full

    @property
    def clone(self) -> "TrendsComputeOpts":
        other: TrendsComputeOpts = pu.deep_clone(self, ignores=["pivot_keys_filter"], assign_ignores=False)
        if self.pivot_keys_filter is not None:
            other.pivot_keys_filter = pu.PropertyValueMaskingOpts(**self.pivot_keys_filter.props)
        return other

    def invalidates_corpus(self, other: "TrendsComputeOpts") -> bool:
        if (
            self.normalize != other.normalize  # pylint: disable=too-many-boolean-expressions
            or self.keyness != other.keyness
            or self.temporal_key != other.temporal_key
            or self.pivot_keys_id_names != other.pivot_keys_id_names
            or self.pivot_keys_filter != other.pivot_keys_filter
            or self.fill_gaps != other.fill_gaps
            or self.keyness_source != other.keyness_source
        ):
            return True
        return False


class ITrendsData(abc.ABC):
    def __init__(self, corpus: pc.VectorizedCorpus, n_top: int = 100000, category_column: str = "time_period"):
        self.corpus: pc.VectorizedCorpus = corpus
        self.n_top: int = n_top

        self._compute_options: Dict = None
        self._gof_data: gof.GofData = None

        self._transformed_corpus: pc.VectorizedCorpus = None
        self._trends_opts: TrendsComputeOpts = TrendsComputeOpts(
            normalize=False, keyness=pk.KeynessMetric.TF, temporal_key='year'
        )
        self.category_column: str = category_column

    @abc.abstractmethod
    def _transform_corpus(self, opts: TrendsComputeOpts) -> pc.VectorizedCorpus:
        ...

    @property
    def transformed_corpus(self) -> pc.VectorizedCorpus:
        return self._transformed_corpus

    @property
    def gof_data(self) -> gof.GofData:
        if self._gof_data is None:
            self._gof_data = gof.GofData.compute(self.corpus, n_top=self.n_top)
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

    def get_top_terms(self, n_top: int = 100, kind='token+count') -> pd.DataFrame:
        top_terms = self._transformed_corpus.get_top_terms(category_column=self.category_column, n_top=n_top, kind=kind)
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
        self._trends_opts = TrendsComputeOpts(normalize=False, keyness=pk.KeynessMetric.TF, temporal_key='year')
        self._gof_data = None
        return self


class TrendsData(ITrendsData):
    def __init__(self, corpus: pc.VectorizedCorpus, n_top: int = 100000):
        super().__init__(corpus=corpus, n_top=n_top)

    def _transform_corpus(self, opts: TrendsComputeOpts) -> pc.VectorizedCorpus:

        corpus: pc.VectorizedCorpus = (
            self.corpus.tf_idf()
            if opts.keyness == pk.KeynessMetric.TF_IDF
            else self.corpus.normalize_by_raw_counts()
            if opts.keyness == pk.KeynessMetric.TF_normalized
            else self.corpus
        )

        corpus = corpus.group_by_pivot_keys(
            temporal_key=opts.temporal_key,
            pivot_keys=list(opts.pivot_keys_id_names),
            pivot_keys_filter=opts.pivot_keys_filter,
            document_namer=None,  # FIXME
            fill_gaps=opts.fill_gaps,
            aggregate='sum',
        )

        if opts.normalize:
            corpus = corpus.normalize_by_raw_counts()

        return corpus


class BundleTrendsData(ITrendsData):
    def __init__(self, bundle: Bundle = None, n_top: int = 100000):
        super().__init__(corpus=bundle.corpus, n_top=n_top)
        self.bundle = bundle
        self.keyness_source: pk.KeynessMetricSource = pk.KeynessMetricSource.Full
        self.tf_threshold: int = 1

    def _transform_corpus(self, opts: TrendsComputeOpts) -> pc.VectorizedCorpus:

        transformed_corpus: pc.VectorizedCorpus = self.bundle.keyness_transform(
            opts=ComputeKeynessOpts(
                period_pivot=opts.temporal_key,
                tf_threshold=self.tf_threshold,
                keyness_source=self.keyness_source,
                keyness=opts.keyness,
                pivot_column_name=self.category_column,
                normalize=opts.normalize,
                fill_gaps=opts.fill_gaps,
            )
        )

        return transformed_corpus
