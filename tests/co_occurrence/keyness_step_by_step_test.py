# type: ignore

from pprint import pprint as pp  # pylint: disable=unused-import
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import pytest
import scipy

from penelope.co_occurrence import Bundle, keyness
from penelope.co_occurrence.keyness import ComputeKeynessOpts, KeynessMetric
from penelope.common.keyness import metrics
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.dtm import ttm_legacy

from ..utils import inline_code
from .utils import create_keyness_opts, create_keyness_test_bundle

SIMPLE_CORPUS_ABCDE_3DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c', 'd', 'c', 'e']),
    ('tran_2019_02_test.txt', ['a', 'a', 'c', 'e', 'c', 'd', 'd']),
    ('tran_2019_03_test.txt', ['d', 'e', 'e', 'b']),
]

# pylint: disable=protected-access


@pytest.mark.long_running
def test_step_by_step_llr_compute_corpus_keyness_alternative():
    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS)
    opts: ComputeKeynessOpts = create_keyness_opts(keyness=KeynessMetric.HAL_cwr)

    corpus: VectorizedCorpus = bundle.corpus
    concept_corpus: VectorizedCorpus = bundle.concept_corpus
    token2id: Token2Id = bundle.token2id
    pivot_key: str = opts.pivot_column_name

    with inline_code(source=keyness.compute_weighed_corpus_keyness):
        zero_out_indices: Sequence[int] = corpus.zero_out_by_tf_threshold(3)
        concept_corpus.zero_out_by_indices(zero_out_indices)

        with inline_code(source=keyness.compute_corpus_keyness):
            corpus = corpus.group_by_temporal_key_optimized(
                temporal_key_specifier=opts.temporal_pivot,
                target_column_name=pivot_key,
                fill_gaps=opts.fill_gaps,
                aggregate='sum',
            )  # matrix([[3, 0, 0, 0, 3, 4, 8, 3, 5, 6, 3, 0, 3]])

            rows = []
            cols = []
            data = []
            pairs2token = (
                corpus.vocabs_mapping.get
            )  # {(0, 2): 0, (0, 3): 2, (0, 4): 3, (0, 5): 10, (0, 6): 7, (2, 3): 1, (2, 4): 5, (2, 6): 11, (3, 4): 4, (3, 6): 12, (4, 5): 6, (4, 6): 9, (5, 6): 8}
            for document_id, term_term_matrix in corpus.to_term_term_matrix_stream(token2id):  # 0,
                # matrix([[0, 0, 3, 0, 0, 3, 3],
                #         [0, 0, 0, 0, 0, 0, 0],
                #         [0, 0, 0, 0, 4, 0, 0],
                #         [0, 0, 0, 0, 3, 0, 3],
                #         [0, 0, 0, 0, 0, 8, 6],
                #         [0, 0, 0, 0, 0, 0, 5],
                #         [0, 0, 0, 0, 0, 0, 0]])
                meta_data = corpus.document_index[corpus.document_index.document_id == 0].to_dict('records')[0]
                weights, (w1_ids, w2_ids) = metrics.significance(
                    TTM=term_term_matrix,
                    metric=opts.keyness,
                    normalize=opts.normalize,
                    n_contexts=meta_data['n_documents'],
                    n_words=meta_data['n_tokens'],
                )
                # (array([-279.97270999,  -23.03480975, -120.70153416,  -85.94256279,
                #         -17.99472463, -182.2522578 ,  -20.19035001,  144.74677931]),
                # (array([0, 0, 2, 3, 3, 4, 4, 5], dtype=int32),
                # array([5, 6, 4, 4, 6, 5, 6, 6], dtype=int32)))
                token_ids = (pairs2token(p) for p in zip(w1_ids, w2_ids))
                rows.extend([document_id] * len(weights))
                cols.extend(token_ids)
                data.extend(weights)

            bag_term_matrix = scipy.sparse.csr_matrix(
                (data, (rows, cols)),
                shape=(len(corpus.document_index), len(corpus.token2id)),
                dtype=np.float64,
            )

            llr_corpus = VectorizedCorpus(
                bag_term_matrix=bag_term_matrix,
                token2id=corpus.token2id,
                document_index=corpus.document_index,
            ).remember(vocabs_mapping=corpus.vocabs_mapping)

            assert llr_corpus is not None

    pp(llr_corpus.data.todense())
    # matrix([[ -15,    0,    0,    0,  -22,  -67, -120,  -11,  144,  -19, -18,    0,   -8]])


def test_LEGACY_step_by_step_llr_compute_corpus_keyness():
    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS)
    opts: ComputeKeynessOpts = create_keyness_opts(keyness=KeynessMetric.LLR)

    corpus: VectorizedCorpus = bundle.corpus
    concept_corpus: VectorizedCorpus = bundle.concept_corpus
    token2id: Token2Id = bundle.token2id
    pivot_key: str = opts.pivot_column_name

    with inline_code(source=keyness.compute_weighed_corpus_keyness):
        zero_out_indices: Sequence[int] = corpus.zero_out_by_tf_threshold(3)
        concept_corpus.zero_out_by_indices(zero_out_indices)

        with inline_code(source=keyness.compute_corpus_keyness):
            corpus = corpus.group_by_temporal_key_optimized(
                temporal_key_specifier=opts.temporal_pivot,
                target_column_name=pivot_key,
                fill_gaps=opts.fill_gaps,
                aggregate='sum',
            )  # matrix([[3, 0, 0, 0, 3, 4, 8, 3, 5, 6, 3, 0, 3]])

            """Current implementation"""
            with inline_code(source=ttm_legacy.LegacyCoOccurrenceMixIn.to_keyness_co_occurrence_corpus):
                with inline_code(source=ttm_legacy.LegacyCoOccurrenceMixIn.to_keyness_co_occurrences):
                    co_occurrences: pd.DataFrame = corpus.to_co_occurrences(token2id)

                    with inline_code(source=metrics.partitioned_significances):
                        vocabulary_size: int = len(token2id)
                        co_occurrence_partitions = []
                        for period in co_occurrences[pivot_key].unique():
                            pivot_co_occurrences = co_occurrences[co_occurrences[pivot_key] == period]
                            term_term_matrix = scipy.sparse.csc_matrix(
                                (pivot_co_occurrences.value, (pivot_co_occurrences.w1_id, pivot_co_occurrences.w2_id)),
                                shape=(vocabulary_size, vocabulary_size),
                                dtype=np.float64,
                            )

                            n_contexts = metrics._get_documents_count(corpus.document_index, pivot_co_occurrences)
                            weights, (w1_ids, w2_ids) = metrics.significance(
                                TTM=term_term_matrix,
                                metric=opts.keyness,
                                normalize=opts.normalize,
                                n_contexts=n_contexts,
                            )
                            co_occurrence_partitions.append(
                                pd.DataFrame(
                                    data={pivot_key: period, 'w1_id': w1_ids, 'w2_id': w2_ids, 'value': weights}
                                )
                            )
                        keyness_co_occurrences = pd.concat(co_occurrence_partitions, ignore_index=True)

                    mg = corpus.get_token_ids_2_pair_id(token2id=token2id).get

                    keyness_co_occurrences['token_id'] = [
                        mg((x[0].item(), x[1].item()))
                        for x in keyness_co_occurrences[['w1_id', 'w2_id']].to_records(index=False)
                    ]

                with inline_code(source=ttm_legacy.LegacyCoOccurrenceMixIn._to_co_occurrence_matrix):
                    pg: Callable = {v: k for k, v in corpus.document_index[pivot_key].to_dict().items()}.get
                    llr_matrix: scipy.sparse.spmatrix = scipy.sparse.coo_matrix(
                        (
                            keyness_co_occurrences.value,
                            (
                                keyness_co_occurrences[pivot_key].apply(pg).astype(np.int32),
                                keyness_co_occurrences.token_id.astype(np.int32),
                            ),
                        ),
                        shape=corpus.data.shape,
                    )

                llr_corpus: VectorizedCorpus = VectorizedCorpus(
                    bag_term_matrix=llr_matrix,
                    token2id=corpus.token2id,
                    document_index=corpus.document_index,
                    vocabs_mapping=corpus.vocabs_mapping,
                )

    assert llr_corpus is not None
    pp(llr_corpus.data.todense())

    # np.matrix([[-15.9, 0.0, 0.0, 0.0, -22.86, -67.30, -120.7, -11.45, 144.74, -19.12, -18.18, 0.0, -8.31]])


def test_dtm_to_ttm_stream():
    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS)
    corpus: VectorizedCorpus = bundle.corpus
    token2id: Token2Id = bundle.token2id

    """Reconstruct TTM row by row"""
    for _, TTM in corpus.to_term_term_matrix_stream(token2id):
        assert TTM is not None


def test_to_keyness():
    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS)
    opts: ComputeKeynessOpts = create_keyness_opts(keyness=KeynessMetric.LLR)

    corpus: VectorizedCorpus = bundle.corpus
    token2id: Token2Id = bundle.token2id

    corpus = corpus.group_by_temporal_key_optimized(
        temporal_key_specifier=opts.temporal_pivot,
        target_column_name=opts.pivot_column_name,
        fill_gaps=opts.fill_gaps,
        aggregate='sum',
    )

    keyness_corpus: VectorizedCorpus = corpus.to_keyness(token2id, opts)

    assert keyness_corpus is not None
