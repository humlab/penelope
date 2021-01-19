from dataclasses import asdict, dataclass, field
from typing import Dict, List

import pandas as pd
import penelope.common.goodness_of_fit as gof
from penelope.corpus import VectorizedCorpus


@dataclass
class TrendsOpts:

    normalize: bool
    tf_idf: bool
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
        if self.tf_idf != other.tf_idf:
            return True
        if self.group_by != other.group_by:
            return True
        return False


@dataclass
class TrendsData:

    corpus: VectorizedCorpus = None
    corpus_folder: str = None
    corpus_tag: str = None
    compute_options: Dict = None

    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None

    current_trends_opts: TrendsOpts = TrendsOpts(normalize=False, tf_idf=False, group_by='year')
    transformed_corpus: VectorizedCorpus = None

    n_count: int = 25000

    memory: dict = field(default_factory=dict)

    def update(
        self,
        *,
        corpus: VectorizedCorpus = None,
        corpus_folder: str = None,
        corpus_tag: str = None,
        n_count: int = None,
    ) -> "TrendsData":

        if (corpus or self.corpus) is None:
            raise ValueError("TrendsData: Corpus is NOT LOADED!")

        self.n_count = n_count or self.n_count
        self.corpus = (corpus or self.corpus).group_by_year()
        self.corpus_folder = corpus_folder or self.corpus_folder
        self.corpus_tag = corpus_tag or self.corpus_tag

        self.compute_options = VectorizedCorpus.load_options(tag=self.corpus_tag, folder=self.corpus_folder)
        self.goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(
            self.corpus, None, verbose=True, metrics=['l2_norm', 'slope']
        )
        self.most_deviating_overview = gof.compile_most_deviating_words(self.goodness_of_fit, n_count=self.n_count)
        self.most_deviating = gof.get_most_deviating_words(
            self.goodness_of_fit, 'l2_norm', n_count=self.n_count, ascending=False, abs_value=True
        )
        return self

    def remember(self, **kwargs) -> "TrendsData":
        self.memory.update(**kwargs)
        return self

    def get_corpus(self, opts: TrendsOpts) -> VectorizedCorpus:

        if self.transformed_corpus is None:
            self.transformed_corpus = self.corpus

        if self.current_trends_opts.invalidates_corpus(opts):

            transformed_corpus: VectorizedCorpus = self.corpus

            if opts.tf_idf:
                transformed_corpus = transformed_corpus.tf_idf()

            transformed_corpus = transformed_corpus.group_by_period(period=opts.group_by)

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
        top_terms = self.transformed_corpus.get_top_terms(category_column='category', n_count=n_count, kind=kind)
        return top_terms
