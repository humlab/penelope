from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Tuple

import numpy as np
import pandas as pd
import scipy
from penelope.common.keyness import KeynessMetric, compute_hal_cwr_score, partitioned_significances
from penelope.utility.utils import create_instance

from ..token2id import Token2Id
from .interface import IVectorizedCorpusProtocol

if TYPE_CHECKING:
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


class CoOccurrenceVocabularyHelper:
    @staticmethod
    def extract_vocabs_mapping_from_co_occurrences(co_occurrences: pd.DataFrame) -> Mapping[Tuple[int, int], int]:
        """Returns a mapping between source/co-occurrence vocabularies"""

        if 'w1_id' not in co_occurrences.columns or 'token_id' not in co_occurrences.columns:
            raise ValueError("fata: cannot create mapping when word ids are missing")

        vocabs_mapping: Mapping[Tuple[int, int], int] = (
            co_occurrences[["w1_id", "w2_id", "token_id"]]
            .drop_duplicates()
            .set_index(['w1_id', 'w2_id'])
            .token_id.to_dict()
        )

        return vocabs_mapping

    @staticmethod
    def extract_vocabs_mapping_from_vocabs(
        corpus: VectorizedCorpus, token2id: Token2Id
    ) -> Mapping[Tuple[int, int], int]:
        """Creates a map from co-occurrence corpus (word-pairs) to source corpus vocabulay (single words)"""
        mapping = {tuple(map(token2id.get, token.split("/"))): token_id for token, token_id in corpus.token2id.items()}
        return mapping

    @staticmethod
    def create_co_occurrence_vocabulary(co_occurrences: pd.DataFrame, token2id: Token2Id) -> Tuple[dict, pd.Series]:
        """Returns a new vocabulary for word-pairs in `co_occurrences`"""

        to_token = token2id.id2token.get
        token_pairs: pd.DataFrame = co_occurrences[["w1_id", "w2_id"]].drop_duplicates().reset_index(drop=True)
        token_pairs["token_id"] = token_pairs.index
        token_pairs["token"] = token_pairs.w1_id.apply(to_token) + "/" + token_pairs.w2_id.apply(to_token)

        """Create a new vocabulary"""
        vocabulary = token_pairs.set_index("token").token_id.to_dict()

        vocabs_mapping: Mapping[Tuple[int, int], int] = token_pairs.set_index(['w1_id', 'w2_id']).token_id.to_dict()

        return vocabulary, vocabs_mapping


class CoOccurrenceMixIn:
    def to_co_occurrence_vocab_mapping(
        self: IVectorizedCorpusProtocol, source_token2id: Token2Id = None
    ) -> Mapping[Tuple[int, int], int]:
        """Returns cached vocabulary mapping"""
        if "vocabs_mapping" not in self.payload:
            if source_token2id is None:
                raise ValueError("fatal: extract_vocabs_mapping_from_vocabs needs a source vocabulary")
            self.payload["vocabs_mapping"] = CoOccurrenceVocabularyHelper.extract_vocabs_mapping_from_vocabs(
                self, source_token2id
            )
        return self.payload["vocabs_mapping"]

    def to_source_vocab_mapping(
        self: IVectorizedCorpusProtocol, source_token2id: Token2Id = None
    ) -> Mapping[Tuple[int, int], int]:
        if "reversed_vocabs_mapping" not in self.payload:
            self.payload["reversed_vocabs_mapping"] = {
                v: k for k, v in self.to_co_occurrence_vocab_mapping(source_token2id).items()
            }
        return self.payload.get("reversed_vocabs_mapping")

    def remember_vocabs_mapping(self, value: Mapping[Tuple[int, int], int]):
        """Stores mapping between vocabularies"""
        self.payload["vocabs_mapping"] = value

    def to_co_occurrences(
        self: IVectorizedCorpusProtocol, source_token2id: Token2Id, partition_key: str = None
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

        """Add a time period column that can be used as a pivot column"""
        df['time_period'] = self.document_index.loc[df.document_id][partition_key].astype(np.int16).values

        # """Decode w1/w2 token pair"""
        # if self.vocabs_mapping is None:
        #     fg = self.id2token.get

        #     df['token'] = df.token_id.apply(fg)

        #     ws = pd.DataFrame(df.token.str.split('/', 1).tolist(), columns=['w1', 'w2'], index=df.index)
        #     df = df.assign(w1=ws.w1, w2=ws.w2)

        #     # """Decode w1 and w2 tokens using supplied token2id"""
        #     sg = source_token2id.get
        #     df['w1_id'] = df.w1.apply(sg).astype(np.int32)
        #     df['w2_id'] = df.w2.apply(sg).astype(np.int32)

        #     df.drop(columns=["token", "w1", "w2"], inplace=True)

        # else:
        pg = self.to_source_vocab_mapping(source_token2id).get
        df[['w1_id', 'w2_id']] = pd.DataFrame(df.token_id.apply(pg).tolist())

        return df

    @staticmethod
    def from_co_occurrences(
        *, co_occurrences: pd.DataFrame, document_index: pd.DataFrame, token2id: Token2Id
    ) -> Tuple[VectorizedCorpus, Mapping[Tuple[int, int], int]]:
        """Creates a co-occurrence DTM corpus from a co-occurrences data frame.

           A "word-pair token" in the corpus' vocabulary has the form "w1/w2".

           The mapping between the two vocabulary is stored in self.payload['vocabs_mapping]
           The mapping translates identities for (w1,w2) to identity for "w1/w2".


        Args:
            co_occurrences (CoOccurrenceDataFrame): [description]
            document_index (DocumentIndex): [description]
            token2id (Token2Id): source corpus vocabulary

        Returns:
            VectorizedCorpus: The co-occurrence corpus
        """

        if not isinstance(token2id, Token2Id):
            token2id = Token2Id(data=token2id)

        vocabulary, vocabs_mapping = CoOccurrenceVocabularyHelper.create_co_occurrence_vocabulary(
            co_occurrences, token2id
        )

        """Set document_id as unique key for DTM document index """
        document_index = document_index.set_index('document_id', drop=False).rename_axis('').sort_index()

        """Make certain that the matrix gets right shape (to avoid offset errors)"""
        shape = (len(document_index), len(vocabulary))

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
        cls: type = create_instance("penelope.corpus.dtm.vectorized_corpus.VectorizedCorpus")
        corpus: VectorizedCorpus = cls(matrix, token2id=vocabulary, document_index=document_index)

        corpus.remember_vocabs_mapping(vocabs_mapping)

        return corpus

    def to_keyness_co_occurrences(
        self: IVectorizedCorpusProtocol,
        keyness: KeynessMetric,
        token2id: Token2Id,
        pivot_key: str,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Returns co-occurrence data frame with weighed values by significance metrics.

        Keyness values are computed for each partition as specified by pivot_key.

        Note: Corpus must be a co-occurrences corpus!
              Tokens must be of the form "w1/w2".
              Supplied token2id must be vocabulary for single words "w1", "w2", ...

        Args:
            token2id (Token2Id): [description]
            pivot_key (str): [description]

        Returns:
            pd.DataFrame: [description]
        """

        co_occurrences: pd.DataFrame = partitioned_significances(
            self.to_co_occurrences(token2id),
            keyness_metric=keyness,
            pivot_key=pivot_key,
            document_index=self.document_index,
            vocabulary_size=len(token2id),
            normalize=normalize,
        )

        mg = self.to_co_occurrence_vocab_mapping().get

        co_occurrences['token_id'] = co_occurrences[['w1_id', 'w2_id']].apply(lambda x: mg((x[0], x[1])), axis=1)

        return co_occurrences

    def to_keyness_co_occurrence_corpus(
        self: IVectorizedCorpusProtocol,
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
            token2id=token2id,
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

        corpus = self.create_instance(matrix, token2id=token2id)

        return corpus

    def HAL_cwr_corpus(
        self: IVectorizedCorpusProtocol,
        *,
        document_window_counts: scipy.sparse.spmatrix,
        vocabs_mapping: Mapping[Tuple[int, int], int],
    ) -> VectorizedCorpus:
        """Returns a BoW co-occurrence corpus where the values are computed HAL CWR score."""

        nw_x = document_window_counts.todense().astype(np.float)
        nw_xy = self.data  # .copy().astype(np.float)

        nw_cwr: scipy.sparse.spmatrix = compute_hal_cwr_score(nw_xy, nw_x, vocabs_mapping)

        cwr_corpus: "VectorizedCorpus" = self.create_instance(bag_term_matrix=nw_cwr)
        return cwr_corpus

    def create_instance(self, bag_term_matrix: scipy.sparse.spmatrix, token2id: Token2Id = None) -> "VectorizedCorpus":
        corpus_class: type = create_instance("penelope.corpus.dtm.vectorized_corpus.VectorizedCorpus")
        corpus: "VectorizedCorpus" = corpus_class(
            bag_term_matrix=bag_term_matrix,
            token2id=self.token2id,
            document_index=self.document_index,
        )

        vocabs_mapping: Any = self.payload.get("vocabs_mapping")

        if vocabs_mapping is None and token2id is not None:
            vocabs_mapping = self.to_co_occurrence_vocab_mapping(token2id)

        if vocabs_mapping is not None:
            corpus.remember_vocabs_mapping(vocabs_mapping)

        return corpus