from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from penelope.common.keyness import compute_hal_cwr_score, metrics
from penelope.type_alias import VocabularyMapping
from penelope.utility import create_class, deprecated

from ..token2id import Token2Id
from .interface import IVectorizedCorpusProtocol

if TYPE_CHECKING:
    from penelope.co_occurrence import TokenWindowCountMatrix
    from penelope.co_occurrence.keyness import ComputeKeynessOpts

    from .corpus import VectorizedCorpus


class ICoOccurrenceVectorizedCorpusProtocol(IVectorizedCorpusProtocol):
    ...

    @property
    def window_counts(self) -> Optional[TokenWindowCountMatrix]:
        ...

    @window_counts.setter
    def window_counts(self, value: TokenWindowCountMatrix) -> None:
        ...

    @property
    def vocabs_mapping(self) -> Optional[VocabularyMapping]:
        ...

    @vocabs_mapping.setter
    def vocabs_mapping(self, value: VocabularyMapping) -> None:
        ...

    def get_token_ids_2_pair_id(self, token2id: Token2Id) -> Optional[Mapping[Tuple[int, int], int]]:
        ...

    def get_pair_id_2_token_ids(self, token2id: Token2Id = None) -> Mapping[int, Tuple[int, int]]:
        ...

    def to_co_occurrences(self, source_token2id: Token2Id, partition_key: str = None) -> pd.DataFrame:
        ...

    def to_keyness(self, token2id: Token2Id, opts: ComputeKeynessOpts) -> pd.DataFrame:
        ...

    def to_HAL_cwr_keyness(self) -> VectorizedCorpus:
        ...

    def create_co_occurrence_corpus(
        self, bag_term_matrix: scipy.sparse.spmatrix, token2id: Token2Id = None
    ) -> "VectorizedCorpus":
        ...


class CoOccurrenceMixIn:
    @property
    def pair_token2id(self: ICoOccurrenceVectorizedCorpusProtocol) -> Token2Id:
        """Alias for token2id for co-occurrence corpus 8where a `token` is actually a token pair)"""
        return self.token2id

    @property
    def window_counts(self: ICoOccurrenceVectorizedCorpusProtocol) -> Optional[TokenWindowCountMatrix]:
        """ Token window count statistics collected during co-occurrence computation"""
        matrix: TokenWindowCountMatrix = self.payload.get("window_counts")
        return matrix

    @window_counts.setter
    def window_counts(self: ICoOccurrenceVectorizedCorpusProtocol, value: TokenWindowCountMatrix) -> None:
        self.remember(window_counts=value)

    @property
    def vocabs_mapping(self: ICoOccurrenceVectorizedCorpusProtocol) -> Optional[VocabularyMapping]:
        """ Translation between single word and word pair vocabularies"""
        return self.payload.get("vocabs_mapping")

    @vocabs_mapping.setter
    def vocabs_mapping(self: ICoOccurrenceVectorizedCorpusProtocol, value: VocabularyMapping) -> None:
        self.remember(vocabs_mapping=value)

    def get_token_ids_2_pair_id(self: ICoOccurrenceVectorizedCorpusProtocol, token2id: Token2Id) -> VocabularyMapping:
        """Returns cached vocabulary mapping"""
        if "vocabs_mapping" not in self.payload:
            if token2id is None:
                raise ValueError("fatal: extract_vocabs_mapping_from_vocabs needs a source vocabulary")
            self.remember(vocabs_mapping=CoOccurrenceVocabularyHelper.extract_pair2token2id_mapping(self, token2id))
        return self.payload.get("vocabs_mapping")

    def get_pair_id_2_token_ids(
        self: ICoOccurrenceVectorizedCorpusProtocol, token2id: Token2Id = None
    ) -> Mapping[int, Tuple[int, int]]:
        if "reversed_vocabs_mapping" not in self.payload:
            self.remember(reversed_vocabs_mapping={v: k for k, v in self.get_token_ids_2_pair_id(token2id).items()})
        return self.payload.get("reversed_vocabs_mapping")

    def to_co_occurrences(
        self: ICoOccurrenceVectorizedCorpusProtocol, token2id: Token2Id, partition_key: str = None
    ) -> pd.DataFrame:
        """Creates a co-occurrence data frame from a vectorized self (DTM)

        source_token2id [Token2Id]: Vocabulary for source corpus
        self.token2id [dict]:       Vocabulary of co-occuring token pairs
        """

        partition_key = partition_key or ('time_period' if 'time_period' in self.document_index.columns else 'year')

        if 0 in self.data.shape:
            return self.empty_data()

        coo = self.data.tocoo(copy=False)
        df = pd.DataFrame(
            {
                'document_id': coo.row.astype(np.int32),
                'token_id': coo.col.astype(np.int32),
                'value': coo.data,
            }
        )

        if len(df) == 0:
            return self.empty_data()

        """Add a time period column that can be used as a pivot column"""
        df['time_period'] = self.document_index.loc[df.document_id][partition_key].astype(np.int16).values

        pg = self.get_pair_id_2_token_ids(token2id).get

        df[['w1_id', 'w2_id']] = pd.DataFrame(df.token_id.apply(pg).tolist())

        return df

    def to_term_term_matrix_stream(
        self: IVectorizedCorpusProtocol, token2id: Token2Id
    ) -> Tuple[int, Iterable[scipy.sparse.spmatrix]]:

        # if USE_NUMBA:
        #     token2pairs: dict = {token_id: pair_ids for pair_ids, token_id in self.vocabs_mapping.items()}
        #     vocab_size: int = len(token2id)

        #     for document_id, term_term_matrix in numba_to_term_term_matrix_stream(
        #         self.data, token2pairs=token2pairs, vocab_size=vocab_size
        #     ):
        #         yield document_id, term_term_matrix
        # else:
        """Generates a sequence of term-term matrices for each document (row)"""
        token2pairs: dict = {token_id: pair_ids for pair_ids, token_id in self.vocabs_mapping.items()}
        """Reconstruct ttm row by row"""
        for i in range(0, self.shape[0]):
            document: scipy.sparse.spmatrix = self.data[i, :]
            if len(document.data) == 0:
                yield i, None
            else:
                rows, cols = zip(*(token2pairs[i] for i in document.indices))
                term_term_matrix: scipy.sparse.spmatrix = scipy.sparse.csc_matrix(
                    (document.data, (rows, cols)), shape=(len(token2id), len(token2id)), dtype=self.data.dtype
                )
                yield i, term_term_matrix

    def to_keyness(self: IVectorizedCorpusProtocol, token2id: Token2Id, opts: ComputeKeynessOpts):
        """Apply keyness to corpus. Return new corpus."""
        rows, cols, data = [], [], []
        pairs2token = self.vocabs_mapping.get
        for document_id, term_term_matrix in self.to_term_term_matrix_stream(token2id):
            if term_term_matrix is None:
                continue
            doc_info: dict = self.document_index[self.document_index.document_id == document_id].to_dict('records')[0]
            weights, (w1_ids, w2_ids) = metrics.significance(
                TTM=term_term_matrix,
                metric=opts.keyness,
                normalize=opts.normalize,
                n_contexts=doc_info.get('n_documents'),
                n_words=doc_info.get('n_raw_tokens', doc_info.get('n_tokens')),
            )
            token_ids = (pairs2token(p) for p in zip(w1_ids, w2_ids))
            rows.extend([document_id] * len(weights))
            cols.extend(token_ids)
            data.extend(weights)

        bag_term_matrix = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(self.document_index), len(self.token2id)), dtype=np.float64
        )

        keyness_corpus = self.corpus_class(
            bag_term_matrix=bag_term_matrix,
            token2id=self.token2id,
            document_index=self.document_index,
        ).remember(vocabs_mapping=self.vocabs_mapping)

        return keyness_corpus

    def to_HAL_cwr_keyness(self: ICoOccurrenceVectorizedCorpusProtocol) -> VectorizedCorpus:
        """Returns a BoW co-occurrence corpus where the values are computed HAL CWR score."""

        if self.window_counts is None:
            raise ValueError("to_HAL_cwr_keyness: payload `window_counts` cannot be empty!")

        if self.vocabs_mapping is None:
            raise ValueError("to_HAL_cwr_keyness: payload `vocabs_mapping` cannot be empty!")

        document_window_counts: scipy.sparse.spmatrix = self.window_counts.document_term_window_counts

        nw_x = document_window_counts
        nw_xy = self.data

        nw_cwr: scipy.sparse.spmatrix = compute_hal_cwr_score(nw_xy, nw_x, self.vocabs_mapping)

        cwr_corpus: "VectorizedCorpus" = self.corpus_class(
            bag_term_matrix=nw_cwr,
            token2id=self.token2id,
            document_index=self.document_index,
        ).remember(vocabs_mapping=self.vocabs_mapping)

        return cwr_corpus

    @property
    def corpus_class(self) -> type:
        return create_class(self._class_name)

    @staticmethod
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


class CoOccurrenceVocabularyHelper:
    @deprecated
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
    def extract_pair2token2id_mapping(corpus: VectorizedCorpus, token2id: Token2Id) -> Mapping[Tuple[int, int], int]:
        """Creates a map from co-occurrence corpus (word-pairs) to source corpus vocabulay (single words)"""
        mapping = {
            tuple(map(token2id.get, token.split(WORD_PAIR_DELIMITER))): token_id
            for token, token_id in corpus.token2id.items()
        }
        return mapping

    @deprecated
    @staticmethod
    def create_pair2id(co_occurrences: pd.DataFrame, token2id: Token2Id) -> Tuple[dict, Mapping[Tuple[int, int], int]]:
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


WORD_PAIR_DELIMITER = "/"


@deprecated
def to_word_pair_token(w1_id: int, w2_id: int, fg: Callable[[int], str]) -> str:
    w1 = fg(w1_id, '').replace(WORD_PAIR_DELIMITER, '')
    w2 = fg(w2_id, '').replace(WORD_PAIR_DELIMITER, '')
    return f"{w1}{WORD_PAIR_DELIMITER}{w2}"
