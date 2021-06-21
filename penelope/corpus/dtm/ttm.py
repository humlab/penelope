from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from penelope.common.keyness import KeynessMetric, compute_hal_cwr_score, partitioned_significances
from penelope.type_alias import VocabularyMapping
from penelope.utility import create_instance, mark_as_disabled

from ..token2id import Token2Id
from .interface import IVectorizedCorpusProtocol

if TYPE_CHECKING:
    from penelope.co_occurrence import TokenWindowCountStatistics

    from .vectorized_corpus import VectorizedCorpus


def empty_data() -> pd.DataFrame:

    frame: pd.DataFrame = pd.DataFrame(
        data={
            'document_id': pd.Series(data=[], dtype=np.int32),
            'token_id': pd.Series(data=[], dtype=np.int32),
            'value': pd.Series(data=[], dtype=np.int32),
            'time_period': pd.Series(data=[], dtype=np.int32),
            'w1_id': pd.Series(data=[], dtype=np.int32),
            'w2_id': pd.Series(data=[], dtype=np.int32),
        }
    )
    return frame


WORD_PAIR_DELIMITER = "/"


def to_word_pair_token(w1_id: int, w2_id: int, fg: Callable[[int], str]) -> str:
    w1 = fg(w1_id, '').replace(WORD_PAIR_DELIMITER, '')
    w2 = fg(w2_id, '').replace(WORD_PAIR_DELIMITER, '')
    return f"{w1}{WORD_PAIR_DELIMITER}{w2}"


class CoOccurrenceVocabularyHelper:
    @staticmethod
    def extract_vocabs_mapping_from_co_occurrences(co_occurrences: pd.DataFrame) -> Mapping[Tuple[int, int], int]:
        """Returns a mapping between source vocabulary and co-occurrence vocabulary"""

        if 'w1_id' not in co_occurrences.columns or 'token_id' not in co_occurrences.columns:
            raise ValueError("fatal: cannot create mapping when word ids are missing")

        vocabs_mapping: Mapping[Tuple[int, int], int] = (
            co_occurrences[["w1_id", "w2_id", "token_id"]]
            .drop_duplicates()
            .set_index(['w1_id', 'w2_id'])
            .token_id.to_dict()
        )

        return vocabs_mapping

    @staticmethod
    def extract_pair_to_single_vocabulary_mapping(
        corpus: VectorizedCorpus, token2id: Token2Id
    ) -> Mapping[Tuple[int, int], int]:
        """Creates a map from co-occurrence corpus (word-pairs) to source corpus vocabulay (single words)"""
        mapping = {
            tuple(map(token2id.get, token.split(WORD_PAIR_DELIMITER))): token_id
            for token, token_id in corpus.token2id.items()
        }
        return mapping

    @staticmethod
    def create_pair_vocabulary(
        co_occurrences: pd.DataFrame, token2id: Token2Id
    ) -> Tuple[dict, Mapping[Tuple[int, int], int]]:
        """Returns a new vocabulary for word-pairs in `co_occurrences`"""

        to_token = lambda x: token2id.id2token.get(x, '').replace(WORD_PAIR_DELIMITER, '')
        token_pairs: pd.DataFrame = co_occurrences[["w1_id", "w2_id"]].drop_duplicates().reset_index(drop=True)
        token_pairs["token_id"] = token_pairs.index
        token_pairs["token"] = (
            token_pairs.w1_id.apply(to_token) + WORD_PAIR_DELIMITER + token_pairs.w2_id.apply(to_token)
        )

        """Create a new vocabulary"""
        vocabulary = token_pairs.set_index("token").token_id.to_dict()

        vocabs_mapping: Mapping[Tuple[int, int], int] = token_pairs.set_index(['w1_id', 'w2_id']).token_id.to_dict()

        return vocabulary, vocabs_mapping


class ICoOccurrenceVectorizedCorpusProtocol(IVectorizedCorpusProtocol):
    ...

    @property
    def window_counts(self) -> Optional[TokenWindowCountStatistics]:
        ...

    @window_counts.setter
    def window_counts(self, value: TokenWindowCountStatistics) -> None:
        ...

    @property
    def vocabs_mapping(self) -> Optional[VocabularyMapping]:
        ...

    @vocabs_mapping.setter
    def vocabs_mapping(self, value: TokenWindowCountStatistics) -> None:
        ...

    def get_pair_vocabulary_mapping(self, single_vocabulary: Token2Id) -> VocabularyMapping:
        ...

    def get_single_vocabulary_mapping(self, single_vocabulary: Token2Id = None) -> Mapping[int, Tuple[int, int]]:
        ...

    def to_co_occurrences(self, source_token2id: Token2Id, partition_key: str = None) -> pd.DataFrame:
        ...

    def to_keyness_co_occurrences(
        self,
        keyness: KeynessMetric,
        single_vocabulary: Token2Id,
        pivot_key: str,
        normalize: bool = False,
    ) -> pd.DataFrame:
        ...

    def to_keyness_co_occurrence_corpus(
        self,
        keyness: KeynessMetric,
        token2id: Token2Id,
        pivot_key: str,
        normalize: bool = False,
    ) -> VectorizedCorpus:
        ...

    def HAL_cwr_corpus(self) -> VectorizedCorpus:
        ...

    def create_co_occurrence_corpus(
        self, bag_term_matrix: scipy.sparse.spmatrix, single_vocabulary: Token2Id = None
    ) -> "VectorizedCorpus":
        ...


class CoOccurrenceMixIn:
    @property
    def window_counts(self: ICoOccurrenceVectorizedCorpusProtocol) -> Optional[TokenWindowCountStatistics]:
        """ Token window count statistics collected during co-occurrence computation"""
        return self.payload.get("window_counts")

    @window_counts.setter
    def window_counts(self: ICoOccurrenceVectorizedCorpusProtocol, value: TokenWindowCountStatistics) -> None:
        self.remember(window_counts=value)

    @property
    def vocabs_mapping(self: ICoOccurrenceVectorizedCorpusProtocol) -> Optional[VocabularyMapping]:
        """ Translation between single word and word pair vocabularies"""
        return self.payload.get("vocabs_mapping")

    @vocabs_mapping.setter
    def vocabs_mapping(self: ICoOccurrenceVectorizedCorpusProtocol, value: TokenWindowCountStatistics) -> None:
        self.remember(vocabs_mapping=value)

    def get_pair_vocabulary_mapping(
        self: ICoOccurrenceVectorizedCorpusProtocol, single_vocabulary: Token2Id
    ) -> VocabularyMapping:
        """Returns cached vocabulary mapping"""
        if "vocabs_mapping" not in self.payload:
            if single_vocabulary is None:
                raise ValueError("fatal: extract_vocabs_mapping_from_vocabs needs a source vocabulary")
            self.remember(
                vocabs_mapping=CoOccurrenceVocabularyHelper.extract_pair_to_single_vocabulary_mapping(
                    self, single_vocabulary
                )
            )
        return self.payload.get("vocabs_mapping")

    def get_single_vocabulary_mapping(
        self: ICoOccurrenceVectorizedCorpusProtocol, single_vocabulary: Token2Id = None
    ) -> Mapping[int, Tuple[int, int]]:
        if "reversed_vocabs_mapping" not in self.payload:
            self.remember(
                reversed_vocabs_mapping={v: k for k, v in self.get_pair_vocabulary_mapping(single_vocabulary).items()}
            )
        return self.payload.get("reversed_vocabs_mapping")

    def to_co_occurrences(
        self: ICoOccurrenceVectorizedCorpusProtocol, source_token2id: Token2Id, partition_key: str = None
    ) -> pd.DataFrame:
        """Creates a co-occurrence data frame from a vectorized self (DTM)

        NOTE:
            source_token2id [Token2Id]: Vocabulary for source corpus
            self.token2id [dict]:       Vocabulary of co-occuring token pairs
        """

        partition_key = partition_key or ('time_period' if 'time_period' in self.document_index.columns else 'year')

        if 0 in self.data.shape:
            return empty_data()

        coo = self.data.tocoo(copy=False)
        df = pd.DataFrame(
            {
                # 'document_id': coo.row,
                'document_id': coo.row.astype(np.int32),
                'token_id': coo.col.astype(np.int32),
                'value': coo.data,
            }
        )

        if len(df) == 0:
            return empty_data()

        """Add a time period column that can be used as a pivot column"""
        df['time_period'] = self.document_index.loc[df.document_id][partition_key].astype(np.int16).values

        pg = self.get_single_vocabulary_mapping(source_token2id).get

        df[['w1_id', 'w2_id']] = pd.DataFrame(df.token_id.apply(pg).tolist())

        # items = df.token_id.apply(pg).tolist()
        # if len(items) != len(df):
        #     logger.warning("len(items) != len(df)")
        # try:
        #     df[['w1_id', 'w2_id']] = pd.DataFrame(items)
        # except Exception as ex:
        #     logger.exception(ex)
        #     logger.info("trying to recover...")
        #     df['w1_id'] = [x[0] if len(x) == 2 else np.NaN for x in items]
        #     df['w2_id'] = [x[1] if len(x) == 2 else np.NaN for x in items]

        return df

    @mark_as_disabled
    @staticmethod
    def from_co_occurrences(
        *, co_occurrences: pd.DataFrame, document_index: pd.DataFrame, token2id: Token2Id
    ) -> Tuple[VectorizedCorpus, Mapping[Tuple[int, int], int]]:
        """Creates a co-occurrence DTM corpus from a co-occurrences data frame.

           A "word-pair token" in the corpus' vocabulary has the form "w1 WORD_PAIR_DELIMITER w2".

           The mapping between the two vocabulary is stored in self.payload['vocabs_mapping]
           The mapping translates identities for (w1,w2) to identity for "w1 WORD_PAIR_DELIMITER w2".


        Args:
            co_occurrences (CoOccurrenceDataFrame): [description]
            document_index (DocumentIndex): [description]
            token2id (Token2Id): source corpus vocabulary

        Returns:
            VectorizedCorpus: The co-occurrence corpus
        """

        if not isinstance(token2id, Token2Id):
            token2id = Token2Id(data=token2id)

        vocabulary, vocabs_mapping = CoOccurrenceVocabularyHelper.create_pair_vocabulary(co_occurrences, token2id)

        """Set document_id as unique key for DTM document index """
        document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

        """Make certain that the matrix gets right shape (to avoid offset errors)"""

        shape = (len(document_index), len(vocabulary))

        if len(vocabulary) == 0:
            matrix = scipy.sparse.coo_matrix(([], (co_occurrences.document_id, [])), shape=shape)
        else:
            fg = vocabs_mapping.get
            matrix = scipy.sparse.coo_matrix(
                (
                    co_occurrences.value.astype(np.int32),
                    (
                        co_occurrences.document_id.astype(np.int32),
                        co_occurrences[['w1_id', 'w2_id']].apply(lambda x: fg((x[0], x[1])), axis=1),
                    ),
                ),
                shape=shape,
            )

        """Create the final corpus (dynamically to avoid cyclic dependency)"""
        corpus_cls: type = create_instance("penelope.corpus.dtm.vectorized_corpus.VectorizedCorpus")
        corpus: VectorizedCorpus = corpus_cls(
            bag_term_matrix=matrix,
            token2id=vocabulary,
            document_index=document_index,
        )

        corpus.remember(vocabs_mapping=vocabs_mapping)

        return corpus

    def to_keyness_co_occurrences(
        self: ICoOccurrenceVectorizedCorpusProtocol,
        keyness: KeynessMetric,
        single_vocabulary: Token2Id,
        pivot_key: str,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Returns co-occurrence data frame with weighed values by significance metrics.

        Keyness values are computed for each partition as specified by pivot_key.

        Note: Corpus must be a co-occurrences corpus!
              Tokens must be of the form "w1 WORD_PAIR_DELIMITER w2".
              Supplied token2id must be vocabulary for single words "w1", "w2", ...

        Args:
            token2id (Token2Id): [description]
            pivot_key (str): [description]

        Returns:
            pd.DataFrame: [description]
        """

        co_occurrences: pd.DataFrame = partitioned_significances(
            self.to_co_occurrences(single_vocabulary),
            keyness_metric=keyness,
            pivot_key=pivot_key,
            document_index=self.document_index,
            vocabulary_size=len(single_vocabulary),
            normalize=normalize,
        )

        mg = self.get_pair_vocabulary_mapping(single_vocabulary=single_vocabulary).get

        co_occurrences['token_id'] = co_occurrences[['w1_id', 'w2_id']].apply(lambda x: mg((x[0], x[1])), axis=1)

        return co_occurrences

    def to_keyness_co_occurrence_corpus(
        self: ICoOccurrenceVectorizedCorpusProtocol,
        keyness: KeynessMetric,
        token2id: Token2Id,
        pivot_key: str,
        normalize: bool = False,
    ) -> VectorizedCorpus:
        """Returns a copy of the corpus where the values have been weighed by keyness metric.

        NOTE: Call only valid for co-occurrence corpus!

        Args:
            token2id (Token2Id): [description]
            pivot_key (str): [description]
            shape (Tuple[int, int]): [description]

        Returns:
            pd.DataFrame: [description]
        """

        co_occurrences: pd.DataFrame = self.to_keyness_co_occurrences(
            keyness=keyness,
            single_vocabulary=token2id,
            pivot_key=pivot_key,
            normalize=normalize,
        )

        """Map that translate pivot_key to document_id"""
        pg = {v: k for k, v in self.document_index[pivot_key].to_dict().items()}.get

        matrix = scipy.sparse.coo_matrix(
            (
                co_occurrences.value,
                (
                    co_occurrences[pivot_key].apply(pg).astype(np.int32),
                    co_occurrences.token_id.astype(np.int32),
                ),
            ),
            shape=self.data.shape,
        )

        corpus = self.create_co_occurrence_corpus(matrix, single_vocabulary=token2id)

        return corpus

    def HAL_cwr_corpus(self: ICoOccurrenceVectorizedCorpusProtocol) -> VectorizedCorpus:
        """Returns a BoW co-occurrence corpus where the values are computed HAL CWR score."""

        if self.window_counts is None:
            raise ValueError("HAL_cwr_corpus: payload `window_counts` cannot be empty!")

        if self.vocabs_mapping is None:
            raise ValueError("HAL_cwr_corpus: payload `vocabs_mapping` cannot be empty!")

        document_window_counts: scipy.sparse.spmatrix = self.window_counts.document_counts

        #  FIXME: cannot do todense if large corpus:
        nw_x = document_window_counts  # .todense().astype(np.float)
        nw_xy = self.data  # .copy().astype(np.float)

        nw_cwr: scipy.sparse.spmatrix = compute_hal_cwr_score(nw_xy, nw_x, self.vocabs_mapping)

        cwr_corpus: "VectorizedCorpus" = self.create_co_occurrence_corpus(bag_term_matrix=nw_cwr)
        return cwr_corpus

    def create_co_occurrence_corpus(
        self, bag_term_matrix: scipy.sparse.spmatrix, single_vocabulary: Token2Id = None
    ) -> "VectorizedCorpus":
        corpus_class: type = create_instance("penelope.corpus.dtm.vectorized_corpus.VectorizedCorpus")
        corpus: "VectorizedCorpus" = corpus_class(
            bag_term_matrix=bag_term_matrix,
            token2id=self.token2id,
            document_index=self.document_index,
        )

        vocabs_mapping: Any = self.payload.get("vocabs_mapping")

        if vocabs_mapping is None and single_vocabulary is not None:
            vocabs_mapping = self.get_pair_vocabulary_mapping(single_vocabulary)

        if vocabs_mapping is not None:
            corpus.remember(vocabs_mapping=vocabs_mapping)

        return corpus
