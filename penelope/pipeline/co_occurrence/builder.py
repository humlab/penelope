import sys
from dataclasses import dataclass
from pprint import pformat as pf
from typing import Mapping, Tuple

import scipy
import scipy.sparse as sp
from loguru import logger
from penelope.co_occurrence import TokenWindowCountMatrix, VectorizedTTM, VectorizeType
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.dtm import WORD_PAIR_DELIMITER
from penelope.type_alias import DocumentIndex

sj = WORD_PAIR_DELIMITER.join

DEBUG_TRACE: bool = False

if DEBUG_TRACE:
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO", enqueue=True)
    logger.add("co_occurrence_trace.py", rotation=None, format="{message}", serialize=False, level="INFO", enqueue=True)


@dataclass
class CoOccurrencePayload:
    document_id: int
    document_name: str
    ttm_data_map: Mapping[VectorizeType, VectorizedTTM]


class CoOccurrenceCorpusBuilder:
    """Creates incrementally a DTM co-occurrence corpus from a stream of (document) TTM matrices"""

    def __init__(
        self,
        vectorize_type: VectorizeType,
        document_index: DocumentIndex,
        pair2id: Token2Id,
        token2id: Token2Id,
    ):
        self.vectorize_type: VectorizeType = vectorize_type
        self.document_index: DocumentIndex = document_index
        self.pair2id: Token2Id = pair2id
        self.token2id: Token2Id = token2id

        """ Co-occurrence DTM matrix """
        self.matrix: sp.spmatrix = None
        self.row = []
        self.col = []
        self.data = []

        """ Per document term window counts """
        self.dtw_counts_row = []
        self.dtw_counts_col = []
        self.dtw_counts_data = []

    # def ingest(self, payloads: Iterable[CoOccurrencePayload]) -> "CoOccurrenceCorpusBuilder":
    #     for payload in payloads:
    #         self.add(payload)
    #     return self

    def add(self, payload: CoOccurrencePayload) -> None:
        """Adds payload to the DTM-data under construction.
        Note! Assumes that data has been ingested to both vocabulary and vocabs mapping."""

        item: VectorizedTTM = payload.ttm_data_map.get(self.vectorize_type)

        if DEBUG_TRACE:
            item.trace()

        pair2id: Token2Id = self.pair2id

        """Translate token-pair ids into new COO-vocabulary ids"""
        TTM: scipy.sparse.spmatrix = item.term_term_matrix
        pair_ids = (pair2id[p] for p in zip(TTM.row, TTM.col))

        self.row.extend([item.document_id] * len(TTM.row))
        self.col.extend(pair_ids)
        self.data.extend(TTM.data)

        """ Add term windows counts """
        counts: Mapping[int, int] = item.term_window_counts

        self.dtw_counts_row.extend([item.document_id] * len(counts))
        self.dtw_counts_col.extend(counts.keys())
        self.dtw_counts_data.extend(counts.values())

        if DEBUG_TRACE:
            self.trace(f"Document ID {item.document_id}")

    @property
    def corpus(self) -> VectorizedCorpus:
        shape: Tuple[int, int] = (len(self.document_index), len(self.pair2id))
        self.matrix = sp.coo_matrix((self.data, (self.row, self.col)), shape=shape)
        corpus: VectorizedCorpus = VectorizedCorpus(
            bag_term_matrix=self.matrix.tocsr(),
            token2id=dict(self.pair2id.data),
            document_index=self.document_index.set_index('document_id', drop=False),
        )

        return corpus

    def compile_window_count_matrix(self) -> TokenWindowCountMatrix:
        window_count_matrix: sp.spmatrix = sp.coo_matrix(
            (self.dtw_counts_data, (self.dtw_counts_row, self.dtw_counts_col))
        ).tocsr()
        matrix: TokenWindowCountMatrix = TokenWindowCountMatrix(document_term_window_counts=window_count_matrix)
        return matrix

    def ingest_pairs(self, payload: CoOccurrencePayload) -> "CoOccurrenceCorpusBuilder":
        """Ingests tokens into pair-vocabulary.
        Note: Tokens are at this stage ingested as `integer tuple pairs` i.e. (w1_id, w2_id) instead of tokens string
              The integer pairs are later updated to `w1/w2` string tokens.
              In this way the pair vocabulary keeps a (w1_id, w2_id) to pair_id mapping
        """

        ttm_item: VectorizedTTM = payload.ttm_data_map.get(self.vectorize_type)

        pairs = zip(ttm_item.term_term_matrix.row, ttm_item.term_term_matrix.col)

        self.pair2id.ingest(pairs)

        return self

    def trace(self, msg: str) -> None:
        logger.info(
            "\n#################################################################################################"
        )
        logger.info("# CoOccurrenceCorpusBuilder")
        logger.info(f"# VectorizeType.{self.vectorize_type.name}")
        logger.info(f"# {msg}")
        logger.info(
            "#################################################################################################\n"
        )
        logger.info(f"vectorize_type = VectorizeType.{self.vectorize_type.name}")
        # logger.info(f"co_occurrence_dtm_matrix: {pf(list(zip(self.row, self.col, self.data)))}")
        # logger.info(
        #     f"document_term_windows_counts = {pf(list(zip(self.dtw_counts_row, self.dtw_counts_col, self.dtw_counts_data)))}"
        # )
        logger.info(f"token2id = {pf(dict(self.token2id.data), compact=True, width=200)}")
        logger.info(f"pair2id = {pf(dict(self.pair2id.data), compact=True, width=1000)}")
        logger.info(f"co_occurrence_corpus = {pf(self.corpus.data.todense())}")
        shape = (len(self.document_index), len(self.token2id))
        logger.info(
            f"document_term_windows_counts = {pf( sp.coo_matrix((self.dtw_counts_data, (self.dtw_counts_row, self.dtw_counts_col)), shape=shape).todense())}"
        )
