from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from loguru import logger
from penelope.corpus.dtm.convert import CoOccurrenceVocabularyHelper
from penelope.type_alias import CoOccurrenceDataFrame

from ..corpus import DocumentIndex, Token2Id, VectorizedCorpus
from . import persistence
from .interface import ContextOpts
from .metrics import compute_hal_cwr_score


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

    # FIXME: Move out of class (possible to dtm.convert.CoOccurrenceMixIn)
    def HAL_cwr_corpus(self) -> VectorizedCorpus:
        """Returns a BoW co-occurrence corpus where the values are computed HAL CWR score."""

        nw_x = self.window_counts.document_counts.todense().astype(np.float)
        nw_xy = self.corpus.data  # .copy().astype(np.float)

        nw_cwr: scipy.sparse.spmatrix = compute_hal_cwr_score(nw_xy, nw_x, self.vocabs_mapping)

        cwr_corpus: VectorizedCorpus = VectorizedCorpus(
            bag_term_matrix=nw_cwr,
            token2id=self.corpus.token2id,
            document_index=self.corpus.document_index,
        )
        return cwr_corpus

    # @property
    # def co_occurrence_filename(self) -> str:
    #     return persistence.co_occurrence_filename(self.folder, self.tag)

    # @property
    # def document_index_filename(self) -> str:
    #     return persistence.document_index_filename(self.folder, self.tag)

    # @property
    # def vocabulary_filename(self) -> str:
    #     return persistence.vocabulary_filename(self.folder, self.tag)

    # @property
    # def options_filename(self) -> str:
    #     return replace_extension(self.co_occurrence_filename, 'json')
