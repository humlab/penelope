import abc
from dataclasses import dataclass, field
from typing import Sequence, overload

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
    pivot_keys_id_names: list[str] = field(default_factory=list)
    # FIXME: Decide if it is best to apply filter in `transform` (reduce corpus) or extract (slice corpus)
    filter_opts: pu.PropertyValueMaskingOpts = None
    unstack_tabular: bool = False

    fill_gaps: bool = False
    smooth: bool = None
    top_count: int = None
    words: list[str] = None
    descending: bool = False
    keyness_source: pk.KeynessMetricSource = pk.KeynessMetricSource.Full

    @property
    def clone(self) -> "TrendsComputeOpts":
        other: TrendsComputeOpts = pu.deep_clone(self, ignores=["filter_opts"], assign_ignores=False)
        if self.filter_opts is not None:
            other.filter_opts = pu.PropertyValueMaskingOpts(**self.filter_opts.props)
        return other

    def invalidates_corpus(self, other: "TrendsComputeOpts") -> bool:
        if (
            self.normalize != other.normalize  # pylint: disable=too-many-boolean-expressions
            or self.keyness != other.keyness
            or self.temporal_key != other.temporal_key
            or self.pivot_keys_id_names != other.pivot_keys_id_names
            or self.filter_opts != other.filter_opts
            or self.fill_gaps != other.fill_gaps
            or self.keyness_source != other.keyness_source
        ):
            return True
        return False

    def update(self, **newdata):
        for key, value in newdata.items():
            setattr(self, key, value)


class TabularCompiler:
    def compile(
        self, *, corpus: pc.VectorizedCorpus, temporal_key: str, pivot_keys_id_names: list[str], indices: Sequence[int]
    ) -> pd.DataFrame:
        """Extracts trend vectors for tokens ´indices` and returns a pd.DataFrame."""

        data: dict = {
            **{temporal_key: corpus.document_index[temporal_key]},
            **{key: corpus.document_index[key] for key in pivot_keys_id_names},
            **{corpus.id2token[token_id]: corpus.bag_term_matrix.getcol(token_id).A.ravel() for token_id in indices},
        }

        return pd.DataFrame(data=data)


class TrendsServiceBase(abc.ABC):
    def __init__(self, corpus: pc.VectorizedCorpus = None, n_top: int = 100000):
        self.n_top: int = n_top

        self._transform_opts: TrendsComputeOpts = TrendsComputeOpts(
            normalize=False, keyness=pk.KeynessMetric.TF, temporal_key='year'
        )
        self.tabular_compiler: TabularCompiler = TabularCompiler()

        self._corpus: pc.VectorizedCorpus = None
        self.corpus = corpus

    @property
    def corpus(self) -> pc.VectorizedCorpus:
        return self._corpus

    @corpus.setter
    def corpus(self, corpus: pc.VectorizedCorpus):
        self._corpus = corpus
        self._transformed_corpus = None
        self._gof_data = None

    @abc.abstractmethod
    def _transform_corpus(self, opts: TrendsComputeOpts) -> pc.VectorizedCorpus:
        ...

    @property
    def transformed_corpus(self) -> pc.VectorizedCorpus:
        return self._transformed_corpus

    @property
    def compute_opts(self) -> TrendsComputeOpts:
        return self._transform_opts

    @property
    def gof_data(self) -> gof.GofData:
        if self._gof_data is None:
            self._gof_data = gof.GofData.compute(self.corpus, n_top=self.n_top)
        return self._gof_data

    @overload
    def find_word_indices(self, opts: TrendsComputeOpts = ...) -> list[int]:
        ...

    @overload
    def find_word_indices(self, words: list[str] = ..., top_count: int = ..., descending: bool = ...) -> list[int]:
        ...

    def find_word_indices(
        self,
        opts: TrendsComputeOpts = None,
        words: list[str] = None,
        top_count: int = None,
        descending: bool = None,
    ) -> list[int]:
        """Find matching word indices. If `opts` is provided, it will be used to transform the corpus before finding matches.
            Otherwise the already transformed corpus will be used.
        Args:
            opts (TrendsComputeOpts, optional): transform options. Defaults to None.
            words (list[str], optional): words to find. Can include wildcards and regular expression. Defaults to None.
            top_count (int, optional): max number of words to return. Defaults to None.
            descending (bool, optional): sort order of returned words. Defaults to None.

        Raises:
            TypeError: If neither opts nor words are provided

        Returns:
            list[int]: list of word indicies
        """
        if isinstance(opts, TrendsComputeOpts):
            return self.transform(opts).transformed_corpus.find_matching_words_indices(
                words or opts.words,
                top_count or opts.top_count,
                descending=descending if descending is not None else opts.descending,
            )

        if isinstance(words, list):
            if self._transformed_corpus is None:
                raise ValueError("Corpus is not transformed")

            return self.transformed_corpus.find_matching_words_indices(words, top_count, descending=descending)

        raise TypeError("Either opts or words must be provided")

    @overload
    def find_words(self, opts: TrendsComputeOpts = ...) -> list[str]:
        ...

    @overload
    def find_words(self, words: list[str] = ..., top_count: int = ..., descending: bool = ...) -> list[str]:
        ...

    def find_words(
        self, opts: TrendsComputeOpts = None, words: list[str] = None, top_count: int = None, descending: bool = None
    ) -> list[str]:
        """Find matching words. If `opts` is provided, it will be used to transform the corpus before finding matches.
            Otherwise the already transformed corpus will be used.
        Args:
            opts (TrendsComputeOpts, optional): transform options. Defaults to None.
            words (list[str], optional): words to find. Can include wildcards and regular expression. Defaults to None.
            top_count (int, optional): max number of words to return. Defaults to None.
            descending (bool, optional): sort order of returned words. Defaults to None.

        Raises:
            TypeError: If neither opts nor words are provided

        Returns:
            list[str]: list of words
        """
        if isinstance(opts, TrendsComputeOpts):
            return self.transform(opts).transformed_corpus.find_matching_words(
                words or opts.words, top_count or opts.top_count, descending=descending or opts.descending
            )
        if isinstance(words, list):
            return self.transformed_corpus.find_matching_words(words, top_count, descending=descending)
        raise TypeError("Either opts or words must be provided")

    def get_top_terms(
        self, n_top: int = 100, kind: str = 'token+count', category_column: str = "category"
    ) -> pd.DataFrame:
        top_terms = self._transformed_corpus.get_top_terms(category_column=category_column, n_top=n_top, kind=kind)
        return top_terms

    def transform(self, opts: TrendsComputeOpts) -> "TrendsServiceBase":
        if self._transformed_corpus is not None:
            if not self._transform_opts.invalidates_corpus(opts):
                return self

        self._transformed_corpus = self._transform_corpus(opts)
        if opts is not self._transform_opts:
            self._transform_opts = opts
        self._gof_data = None

        return self

    @overload
    def extract(self, indices: Sequence[int | str] = ..., filter_opts: pu.PropertyValueMaskingOpts = ...) -> list[str]:
        ...

    @overload
    def extract(self, indices: list[str] = ..., top_count: int = ..., descending: bool = ...) -> list[str]:
        ...

    def extract(
        self,
        indices: Sequence[int | str],
        top_count: int = None,
        descending: bool = None,
        filter_opts: pu.PropertyValueMaskingOpts = None,
    ) -> pd.DataFrame:
        if len(indices) > 0:
            if isinstance(indices[0], str):
                indices = self.transformed_corpus.find_matching_words_indices(indices, top_count, descending=descending)

        data: pd.DataFrame = self.tabular_compiler.compile(
            corpus=self.transformed_corpus,
            temporal_key=self.compute_opts.temporal_key,
            pivot_keys_id_names=self.compute_opts.pivot_keys_id_names,
            indices=indices,
        )
        if filter_opts and len(filter_opts) > 0:
            data = data[filter_opts.mask(data)]
        return data

    def reset(self) -> "TrendsServiceBase":
        self._transformed_corpus = None
        self._transform_opts = TrendsComputeOpts(normalize=False, keyness=pk.KeynessMetric.TF, temporal_key='year')
        self._gof_data = None
        return self


class TrendsService(TrendsServiceBase):
    def __init__(self, corpus: pc.VectorizedCorpus = None, n_top: int = 100000):
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
            filter_opts=opts.filter_opts,
            document_namer=None,  # FIXME
            fill_gaps=opts.fill_gaps,
            aggregate='sum',
        )

        if opts.normalize:
            corpus = corpus.normalize_by_raw_counts()

        return corpus


class BundleTrendsService(TrendsServiceBase):
    def __init__(self, bundle: Bundle = None, n_top: int = 100000, category_column: str = 'category'):
        super().__init__(corpus=bundle.corpus, n_top=n_top)
        self.keyness_source: pk.KeynessMetricSource = pk.KeynessMetricSource.Full
        self.tf_threshold: int = 1
        self.category_column: str = category_column

        self._bundle: Bundle = None
        self.bundle = bundle

    @property
    def bundle(self) -> Bundle:
        return self._bundle

    @bundle.setter
    def bundle(self, bundle: Bundle):
        self._bundle = bundle
        self.corpus = bundle.corpus if bundle else None

    def _transform_corpus(self, opts: TrendsComputeOpts) -> pc.VectorizedCorpus:
        transformed_corpus: pc.VectorizedCorpus = self.bundle.keyness_transform(
            opts=ComputeKeynessOpts(
                temporal_pivot=opts.temporal_key,
                tf_threshold=self.tf_threshold,
                keyness_source=self.keyness_source,
                keyness=opts.keyness,
                pivot_column_name=opts.temporal_key,
                normalize=opts.normalize,
                fill_gaps=opts.fill_gaps,
            )
        )

        return transformed_corpus


TrendsDataBase = TrendsServiceBase
BundleTrendsData = BundleTrendsService
TrendsData = TrendsService
