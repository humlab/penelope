# type: ignore

from pprint import pprint as pp  # pylint: disable=unused-import
from typing import Sequence

import numpy as np
import pandas as pd
import scipy
from penelope.co_occurrence import Bundle, keyness
from penelope.co_occurrence.keyness import ComputeKeynessOpts, KeynessMetric
from penelope.common.keyness import metrics
from penelope.corpus import Token2Id, VectorizedCorpus

from ..utils import incline_code
from .utils import create_keyness_opts, create_keyness_test_bundle

SIMPLE_CORPUS_ABCDE_3DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c', 'd', 'c', 'e']),
    ('tran_2019_02_test.txt', ['a', 'a', 'c', 'e', 'c', 'd', 'd']),
    ('tran_2019_03_test.txt', ['d', 'e', 'e', 'b']),
]

# pylint: disable=protected-access


def test_step_by_step_llr_compute_corpus_keyness_alternative():
    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS)
    opts: ComputeKeynessOpts = create_keyness_opts(keyness=KeynessMetric.LLR)

    corpus: VectorizedCorpus = bundle.corpus
    concept_corpus: VectorizedCorpus = bundle.concept_corpus
    token2id: Token2Id = bundle.token2id
    pivot_key: str = opts.pivot_column_name

    with incline_code(source=keyness.compute_weighed_corpus_keyness):

        zero_out_indices: Sequence[int] = corpus.zero_out_by_tf_threshold(3)
        concept_corpus.zero_out_by_indices(zero_out_indices)

        with incline_code(source=keyness.compute_corpus_keyness):

            corpus = corpus.group_by_time_period_optimized(
                time_period_specifier=opts.period_pivot,
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
                n_documents = int(corpus.document_index[corpus.document_index.document_id == 0]['n_documents'])  # 3
                weights, (w1_ids, w2_ids) = metrics.significance(
                    TTM=term_term_matrix,
                    metric=opts.keyness,
                    normalize=opts.normalize,
                    n_contexts=n_documents,
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

    with incline_code(source=keyness.compute_weighed_corpus_keyness):

        zero_out_indices: Sequence[int] = corpus.zero_out_by_tf_threshold(3)
        concept_corpus.zero_out_by_indices(zero_out_indices)

        with incline_code(source=keyness.compute_corpus_keyness):

            corpus = corpus.group_by_time_period_optimized(
                time_period_specifier=opts.period_pivot,
                target_column_name=pivot_key,
                fill_gaps=opts.fill_gaps,
                aggregate='sum',
            )  # matrix([[3, 0, 0, 0, 3, 4, 8, 3, 5, 6, 3, 0, 3]])

            """Current implementation"""
            with incline_code(source=corpus.to_keyness_co_occurrence_corpus):

                with incline_code(source=corpus.to_keyness_co_occurrences):

                    co_occurrences: pd.DataFrame = corpus.to_co_occurrences(token2id)
                    #    document_id  token_id  value  time_period  w1_id  w2_id
                    # 0            0         0      3         2019      0      2
                    # 1            0         4      3         2019      3      4
                    # 2            0         5      4         2019      2      4
                    # 3            0         6      8         2019      4      5
                    # 4            0         7      3         2019      0      6
                    # 5            0         8      5         2019      5      6
                    # 6            0         9      6         2019      4      6
                    # 7            0        10      3         2019      0      5
                    # 8            0        12      3         2019      3      6

                    with incline_code(source=metrics.partitioned_significances):
                        vocabulary_size: int = len(token2id)
                        co_occurrence_partitions = []
                        for period in co_occurrences[pivot_key].unique():
                            pivot_co_occurrences = co_occurrences[co_occurrences[pivot_key] == period]
                            term_term_matrix = scipy.sparse.csc_matrix(
                                (pivot_co_occurrences.value, (pivot_co_occurrences.w1_id, pivot_co_occurrences.w2_id)),
                                shape=(vocabulary_size, vocabulary_size),
                                dtype=np.float64,
                            )
                            # matrix([[0., 0., 3., 0., 0., 3., 3.],
                            #         [0., 0., 0., 0., 0., 0., 0.],
                            #         [0., 0., 0., 0., 4., 0., 0.],
                            #         [0., 0., 0., 0., 3., 0., 3.],
                            #         [0., 0., 0., 0., 0., 8., 6.],
                            #         [0., 0., 0., 0., 0., 0., 5.],
                            #         [0., 0., 0., 0., 0., 0., 0.]])

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

                llr_matrix = corpus._to_co_occurrence_matrix(keyness_co_occurrences, pivot_key)
                llr_corpus = corpus.create_co_occurrence_corpus(llr_matrix, token2id=token2id)

    assert llr_corpus is not None
    pp(llr_corpus.data.todense())

    np.matrix(
        [
            [
                -15.95593619,
                0.0,
                0.0,
                0.0,
                -22.86600166,
                -67.3019315,
                -120.72753604,
                -11.45725503,
                144.74677931,
                -19.12142693,
                -18.1873717,
                0.0,
                -8.31776617,
            ]
        ]
    )


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

    corpus = corpus.group_by_time_period_optimized(
        time_period_specifier=opts.period_pivot,
        target_column_name=opts.pivot_column_name,
        fill_gaps=opts.fill_gaps,
        aggregate='sum',
    )

    keyness_corpus: VectorizedCorpus = corpus.to_keyness(token2id, opts)

    assert keyness_corpus is not None
