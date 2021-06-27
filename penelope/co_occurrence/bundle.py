from typing import List, Mapping, Optional, Set, Tuple

import pandas as pd
from loguru import logger
from penelope.corpus.dtm.ttm import CoOccurrenceVocabularyHelper
from penelope.type_alias import CoOccurrenceDataFrame
from penelope.utility import flatten

from ..corpus import DocumentIndex, Token2Id, VectorizedCorpus
from . import persistence
from .interface import ContextOpts
from .keyness import ComputeKeynessOpts, compute_weighed_corpus_keyness


class Bundle:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        corpus: VectorizedCorpus = None,
        token2id: Token2Id = None,
        document_index: DocumentIndex = None,
        concept_corpus: VectorizedCorpus = None,
        folder: str = None,
        tag: str = None,
        compute_options: dict = None,
        co_occurrences: pd.DataFrame = None,
        vocabs_mapping: Optional[Mapping[Tuple[int, int], int]] = None,
    ):
        """Full co-occurrence corpus where the tokens are concatenated co-occurring word-pairs"""
        self.corpus: VectorizedCorpus = corpus

        """Source corpus vocabulary (i.e. not token-pairs)"""
        self.token2id: Token2Id = token2id
        self.document_index: DocumentIndex = document_index

        self.folder: str = folder
        self.tag: str = tag

        self.compute_options: dict = compute_options

        """Concept context co-occurrence corpus"""
        self.concept_corpus: VectorizedCorpus = concept_corpus

        self._co_occurrences: pd.DataFrame = co_occurrences
        self._token_ids_2_pair_id: Optional[Mapping[Tuple[int, int], int]] = vocabs_mapping

        self.remember_vocabs_mapping()

    def compress(self) -> "Bundle":
        def _token_ids_to_keep(kept_pair_ids: Set[int]) -> List[int]:
            token_ids_in_kept_pairs: Set[int] = set(
                flatten((k for k, pair_id in self.token_ids_2_pair_id.items() if pair_id in kept_pair_ids))
            )
            kept_token_ids: List[int] = sorted(list(token_ids_in_kept_pairs.union(set(self.token2id.magic_token_ids))))
            return kept_token_ids

        if not self.concept_corpus:
            return self

        """Compress concep corpus (remove zero-columns)"""
        _, ids_translation, kept_pair_ids = self.concept_corpus.compress(tf_threshold=1, inplace=True)

        """Slice full corpus to match compressed concept corpus columns"""
        self.corpus.slice_by_indices(kept_pair_ids, inplace=True)

        """Update token count and token2id"""
        kept_token_ids = _token_ids_to_keep(set(kept_pair_ids))

        self.corpus.window_counts.clip(kept_token_ids, inplace=True)
        self.concept_corpus.window_counts.clip(kept_token_ids, inplace=True)

        self.token2id.translate(ids_translation=ids_translation, inplace=True)

        self._token_ids_2_pair_id = {
            pair: pair_id for pair, pair_id in self._token_ids_2_pair_id if pair_id in ids_translation
        }

        return self

    def keyness_transform(self, *, opts: ComputeKeynessOpts) -> VectorizedCorpus:
        """Returns a grouped, keyness adjusted corpus filtered by  threshold."""

        corpus: VectorizedCorpus = compute_weighed_corpus_keyness(
            self.corpus,
            self.concept_corpus,
            token2id=self.token2id,
            opts=opts,
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
    def token_ids_2_pair_id(self) -> Mapping[Tuple[int, int], int]:
        if self._token_ids_2_pair_id is None:
            self._token_ids_2_pair_id = self.corpus.get_token_ids_2_pair_id(self.token2id)
        return self._token_ids_2_pair_id

    @token_ids_2_pair_id.setter
    def token_ids_2_pair_id(self, value: Mapping[Tuple[int, int], int]):
        self._token_ids_2_pair_id = value

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

        bundle: Bundle = persistence.load(
            filename=filename, folder=folder, tag=tag, compute_frame=compute_frame
        ).remember_vocabs_mapping()

        if bundle.co_occurrences is None and compute_frame:
            raise NotImplementedError("THIS IS PROBABLY A BUG")
            # bundle.co_occurrences = bundle.corpus.to_co_occurrences(bundle.token2id)

        return bundle

    def remember_vocabs_mapping(self) -> "Bundle":
        if self.token_ids_2_pair_id is None:
            self.token_ids_2_pair_id = CoOccurrenceVocabularyHelper.extract_pair2token2id_mapping(
                self.corpus, self.token2id
            )

        if self.corpus.vocabs_mapping is None:
            self.corpus.remember(vocabs_mapping=self.token_ids_2_pair_id)

        if self.concept_corpus and self.concept_corpus.vocabs_mapping is None:
            self.concept_corpus.remember(vocabs_mapping=self.token_ids_2_pair_id)

        return self

    @property
    def decoded_co_occurrences(self) -> pd.DataFrame:
        fg = self.token2id.id2token.get
        return self.co_occurrences.assign(
            w1=self.co_occurrences.w1_id.apply(fg),
            w2=self.co_occurrences.w2_id.apply(fg),
        )
