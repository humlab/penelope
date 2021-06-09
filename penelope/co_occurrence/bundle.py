from typing import Mapping, Optional, Tuple, Union

import pandas as pd
from loguru import logger
from penelope.common.keyness import KeynessMetric
from penelope.corpus.dtm.ttm import CoOccurrenceVocabularyHelper
from penelope.type_alias import CoOccurrenceDataFrame

from ..corpus import DocumentIndex, Token2Id, VectorizedCorpus
from . import persistence
from .interface import ContextOpts


class Bundle:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        corpus: VectorizedCorpus = None,
        token2id: Token2Id = None,
        document_index: DocumentIndex = None,
        window_counts: persistence.TokenWindowCountStatistics = None,
        folder: str = None,
        tag: str = None,
        compute_options: dict = None,
        co_occurrences: pd.DataFrame = None,
        vocabs_mapping: Optional[Mapping[Tuple[int, int], int]] = None,
    ):
        self.corpus: VectorizedCorpus = corpus
        self.token2id: Token2Id = token2id
        self.document_index: DocumentIndex = document_index
        self.window_counts: persistence.TokenWindowCountStatistics = window_counts
        self.folder: str = folder
        self.tag: str = tag
        self.compute_options: dict = compute_options

        self._co_occurrences: pd.DataFrame = co_occurrences
        self._vocabs_mapping: Optional[Mapping[Tuple[int, int], int]] = vocabs_mapping

        """Co-occurrence corpus where the tokens are concatenated co-occurring word-pairs"""
        """Source corpus vocabulary (i.e. not token-pairs)"""

    def to_keyness_corpus(
        self,
        period_pivot: str,
        keyness: KeynessMetric,
        global_threshold: Union[int, float],
        pivot_column_name: str,
        normalize: bool = False,
        fill_gaps: bool = False,
    ) -> VectorizedCorpus:
        """Returns a grouped, optionally TF-IDF, corpus filtered by token & threshold.
        Returned corpus' document index has a new pivot column `target_column_name`

        Args:
            corpus (VectorizedCorpus): input corpus
            period_pivot (str): temporal pivot key
            keyness (KeynessMetric): keyness metric to apply on corpus
            token_filter (str): match tokens
            global_threshold (Union[int, float]): limit result by global term frequency
            pivot_column_name (Union[int, float]): name of grouping column

        Returns:
            VectorizedCorpus: pivoted corpus.
        """
        if period_pivot not in ["year", "lustrum", "decade"]:
            raise ValueError(f"illegal time period {period_pivot}")

        corpus: VectorizedCorpus = self.corpus

        if global_threshold > 1:
            corpus = corpus.slice_by_term_frequency(global_threshold)

        """Metrics computed on a document level"""
        if keyness == KeynessMetric.TF_IDF:
            corpus = corpus.tf_idf()
        elif keyness == KeynessMetric.TF_normalized:
            corpus = corpus.normalize_by_raw_counts()
        elif keyness == KeynessMetric.TF:
            pass

        corpus = corpus.group_by_time_period_optimized(
            time_period_specifier=period_pivot,
            target_column_name=pivot_column_name,
            fill_gaps=fill_gaps,
        )

        """Metrics computed on partitioned corpus"""
        if keyness in (KeynessMetric.PPMI, KeynessMetric.LLR, KeynessMetric.DICE, KeynessMetric.LLR_Dunning):
            corpus = corpus.to_keyness_co_occurrence_corpus(
                keyness=keyness,
                token2id=self.token2id,
                pivot_key=pivot_column_name,
                normalize=normalize,
            )
        elif keyness == KeynessMetric.HAL_cwr:
            corpus = corpus.HAL_cwr_corpus(
                document_window_counts=self.window_counts.document_counts,
                vocabs_mapping=self.vocabs_mapping,
            )

        return corpus

    @property
    def co_occurrences(self) -> CoOccurrenceDataFrame:
        if self._co_occurrences is None:
            logger.info("Generating co-occurrences data frame....")
            self._co_occurrences = self.corpus.to_co_occurrences(self.token2id)
        return self._co_occurrences

    @co_occurrences.setter
    def co_occurrences(self, value: pd.DataFrame):
        self._co_occurrences = value

    @property
    def vocabs_mapping(self) -> Mapping[Tuple[int, int], int]:
        if self._vocabs_mapping is None:
            self._vocabs_mapping = self.corpus.to_co_occurrence_vocab_mapping(self.token2id)
        return self._vocabs_mapping

    @vocabs_mapping.setter
    def vocabs_mapping(self, value: Mapping[Tuple[int, int], int]):
        self._vocabs_mapping = value

    @property
    def context_opts(self) -> Optional[ContextOpts]:
        opts: dict = (self.compute_options or dict()).get("context_opts")
        if opts is None:
            return None
        context_opts = ContextOpts.from_kwargs(**opts)
        return context_opts

    def store(self, *, folder: str = None, tag: str = None) -> "Bundle":

        if tag and folder:
            self.tag, self.folder = tag, folder

        persistence.store(self)

        return self

    @staticmethod
    def load(filename: str = None, folder: str = None, tag: str = None, compute_frame: bool = True) -> "Bundle":
        """Loads bundle identified by given filename i.e. `folder`/`tag`{FILENAME_POSTFIX}"""

        data = persistence.load(filename=filename, folder=folder, tag=tag, compute_frame=compute_frame)

        bundle = Bundle(**data)

        if bundle.vocabs_mapping is None:
            bundle.vocabs_mapping = CoOccurrenceVocabularyHelper.extract_vocabs_mapping_from_vocabs(
                bundle.corpus, bundle.token2id
            )

        bundle.corpus.remember_vocabs_mapping(bundle.vocabs_mapping)

        if bundle.co_occurrences is None and compute_frame:
            bundle.co_occurrences = bundle.corpus.to_co_occurrences(bundle.token2id)

        return bundle

    @property
    def decoded_co_occurrences(self) -> pd.DataFrame:
        fg = self.token2id.id2token.get
        return self.co_occurrences.assign(
            w1=self.co_occurrences.w1_id.apply(fg),
            w2=self.co_occurrences.w2_id.apply(fg),
        )
