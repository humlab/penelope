import abc
from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd
from penelope.co_occurrence import Bundle
from penelope.common.goodness_of_fit import GofData
from penelope.common.keyness import KeynessMetric
from penelope.corpus import VectorizedCorpus


@dataclass
class TrendsOpts:

    normalize: bool
    keyness: KeynessMetric
    group_by: str

    fill_gaps: bool = False
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
        if self.fill_gaps != other.fill_gaps:
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
        self._trends_opts: TrendsOpts = TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='year')
        self.category_column: str = "time_period"

    @abc.abstractmethod
    def _transform_corpus(self, opts: TrendsOpts) -> VectorizedCorpus:
        ...

    @property
    def transformed_corpus(self) -> VectorizedCorpus:
        return self._transformed_corpus

    @property
    def gof_data(self) -> GofData:
        if self._gof_data is None:
            self._gof_data = GofData.compute(self.corpus, n_count=self.n_count)
        return self._gof_data

    def find_word_indices(self, opts: TrendsOpts) -> List[int]:
        indices: List[int] = self._transform_corpus(opts).find_matching_words_indices(
            opts.words, opts.word_count, descending=opts.descending
        )
        return indices

    def find_words(self, opts: TrendsOpts) -> List[str]:
        words: List[int] = self._transform_corpus(opts).find_matching_words(
            opts.words, opts.word_count, descending=opts.descending
        )
        return words

    def get_top_terms(self, n_count: int = 100, kind='token+count') -> pd.DataFrame:
        top_terms = self._transformed_corpus.get_top_terms(
            category_column=self.category_column, n_count=n_count, kind=kind
        )
        return top_terms

    def transform(self, opts: TrendsOpts) -> "ITrendsData":

        if self._transformed_corpus is not None:
            if not self._trends_opts.invalidates_corpus(opts):
                return self

        self._transformed_corpus = self._transform_corpus(opts)
        self._trends_opts = opts.clone
        self._gof_data = None

        return self

    def reset(self) -> "ITrendsData":
        self._transformed_corpus = None
        self._trends_opts = TrendsOpts(normalize=False, keyness=KeynessMetric.TF, group_by='year')
        self._gof_data = None
        return self


class TrendsData(ITrendsData):
    def __init__(self, corpus: VectorizedCorpus, corpus_folder: str, corpus_tag: str, n_count: int = 100000):
        super().__init__(corpus=corpus, corpus_folder=corpus_folder, corpus_tag=corpus_tag, n_count=n_count)

    def _transform_corpus(self, opts: TrendsOpts) -> VectorizedCorpus:

        transformed_corpus: VectorizedCorpus = self.corpus

        """ Normal word trends """
        if opts.keyness == KeynessMetric.TF_IDF:
            transformed_corpus = transformed_corpus.tf_idf()
        elif opts.keyness == KeynessMetric.TF_normalized:
            transformed_corpus = transformed_corpus.normalize_by_raw_counts()

        transformed_corpus = transformed_corpus.group_by_time_period(
            time_period_specifier=opts.group_by,
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

    def _transform_corpus(self, opts: TrendsOpts) -> VectorizedCorpus:

        transformed_corpus: VectorizedCorpus = self.bundle.to_keyness_corpus(
            period_pivot=opts.group_by,
            global_threshold=1,
            keyness=opts.keyness,
            pivot_column_name=self.category_column,
            normalize=opts.normalize,
            fill_gaps=opts.fill_gaps,
        )

        return transformed_corpus
