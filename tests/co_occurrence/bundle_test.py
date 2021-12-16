# type: ignore
# pylint: disable=unused-import, unused-variable

from collections import Counter
from pprint import pformat as pf
from pprint import pprint as pp
from typing import Iterable, List, Set

import numpy as np
import pandas as pd
import pytest
from penelope.co_occurrence import Bundle, ContextOpts, TokenWindowCountMatrix
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus.dtm.corpus import VectorizedCorpus
from penelope.utility.utils import flatten

from .utils import create_simple_bundle_by_pipeline

SIMPLE_CORPUS_ABCDE_3DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c', 'd', 'c', 'e']),
    ('tran_2019_02_test.txt', ['a', 'a', 'c', 'e', 'c', 'd', 'd']),
    ('tran_2019_03_test.txt', ['d', 'e', 'e', 'b']),
]
# {0: 1, 1: 1, 2: 3, 3: 2, 4: 5, 5: 4, 6: 4}
# {'c/d': 0, '*/e': 1, 'd/e': 2, 'c/e': 3, '*/d': 4}

# pylint: disable=protected-access,too-many-statements


def test_compress_corpus():

    context_opts: ContextOpts = ContextOpts(
        concept={'d'}, ignore_concept=False, context_width=1, processes=None, ignore_padding=False
    )

    bundle: Bundle = create_simple_bundle_by_pipeline(
        data=SIMPLE_CORPUS_ABCDE_3DOCS, context_opts=context_opts, compress=False
    )
    concept_corpus: VectorizedCorpus = bundle.concept_corpus

    assert (
        (
            concept_corpus.data.todense()
            == np.matrix(
                [
                    [0, 0, 0, 0, 5, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 1, 1, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0],
                ],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )

    _, ids_translation, keep_ids = concept_corpus.compress(tf_threshold=1, extra_keep_ids=[1], inplace=True)

    assert (
        (
            concept_corpus.data.todense()
            == np.matrix(
                [[0, 5, 0, 1, 1, 0], [0, 3, 0, 1, 1, 2], [0, 0, 1, 0, 3, 1]],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )
    assert keep_ids.tolist() == [1, 4, 7, 8, 9, 10]
    assert ids_translation == {1: 0, 4: 1, 7: 2, 8: 3, 9: 4, 10: 5}


@pytest.mark.long_running
def test_step_by_step_compress_with_simple_corpus():

    context_opts: ContextOpts = ContextOpts(concept={'d'}, ignore_concept=False, context_width=1, ignore_padding=False)

    bundle: Bundle = create_simple_bundle_by_pipeline(
        data=SIMPLE_CORPUS_ABCDE_3DOCS, context_opts=context_opts, compress=False
    )

    token2id = dict(bundle.token2id.data)
    assert token2id == {'*': 0, 'd': 1, '__low-tf__': 2, 'a': 3, 'b': 4, 'c': 5, 'e': 6}

    windows = [
        [
            [bundle.token2id.id2token[x] for x in window]
            for window in generate_windows(
                token_ids=[bundle.token2id[t] for t in tokens],
                context_width=context_opts.context_width,
                pad_id=bundle.token2id[context_opts.pad],
                ignore_pads=False,
            )
        ]
        for _, tokens in SIMPLE_CORPUS_ABCDE_3DOCS
    ]
    assert windows == [
        [
            ['*', 'a', 'b'],
            ['a', 'b', 'c'],
            ['b', 'c', 'c'],
            ['c', 'c', 'd'],
            ['c', 'd', 'c'],
            ['d', 'c', 'e'],
            ['c', 'e', '*'],
        ],
        [
            ['*', 'a', 'a'],
            ['a', 'a', 'c'],
            ['a', 'c', 'e'],
            ['c', 'e', 'c'],
            ['e', 'c', 'd'],
            ['c', 'd', 'd'],
            ['d', 'd', '*'],
        ],
        [
            ['*', 'd', 'e'],
            ['d', 'e', 'e'],
            ['e', 'e', 'b'],
            ['e', 'b', '*'],
        ],
    ]

    concept_windows = [[window for window in document if 'd' in window] for document in windows]
    assert concept_windows == [
        [
            ['c', 'c', 'd'],
            ['c', 'd', 'c'],
            ['d', 'c', 'e'],
        ],
        [
            ['e', 'c', 'd'],
            ['c', 'd', 'd'],
            ['d', 'd', '*'],
        ],
        [
            ['*', 'd', 'e'],
            ['d', 'e', 'e'],
        ],
    ]

    co_occurrence_dtm = bundle.concept_corpus.data.todense()
    assert (
        (
            co_occurrence_dtm
            == np.matrix(
                [
                    [0, 0, 0, 0, 5, 0, 0, 0, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 3, 0, 0, 0, 1, 1, 2, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0],
                ],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )

    # id2pair = bundle.corpus.id2token
    # co_occurrence_dtm_frame = pd.DataFrame(co_occurrence_dtm, columns=[id2pair[i] for i in range(0, len(id2pair))])
    # assert co_occurrence_dtm_frame is not None
    """
        print(co_occurrence_dtm_frame)

                             *              *    *    *    *
         0    1    2    3    4    5    6    7    8    9   10   11   12
        */a  a/b  */b  */c  d/c  b/c  a/c  */e  c/e  d/e  */d  a/e  b/e
        --------------------------------------------------------------
    0    0    0    0    0    5    0    0    0    1    1    0    0    0
    1    0    0    0    0    3    0    0    0    1    1    2    0    0
    2    0    0    0    0    0    0    0    1    0    3    1    0    0
        --------------------------------------------------------------
  SUM    0    0    0    0    8    0    0    1    2    5    3    0    0


    """

    """ Compress concept corpus (inlined code) """

    concept_corpus: VectorizedCorpus = bundle.concept_corpus

    # _, ids_translation, kept_pair_ids = concept_corpus.compress(tf_threshold=1, inplace=True)

    extra_keep_ids = []
    keep_ids = concept_corpus.term_frequencies_greater_than_or_equal_to_threshold(1, keep_indices=extra_keep_ids)
    assert keep_ids.tolist() == [4, 7, 8, 9, 10]

    extra_keep_ids = [1]
    keep_ids = concept_corpus.term_frequencies_greater_than_or_equal_to_threshold(1, keep_indices=extra_keep_ids)

    assert keep_ids.tolist() == [1, 4, 7, 8, 9, 10]

    ids_translation = {old_id: new_id for new_id, old_id in enumerate(keep_ids)}
    assert ids_translation == {1: 0, 4: 1, 7: 2, 8: 3, 9: 4, 10: 5}

    concept_corpus.slice_by_indices(keep_ids, inplace=True)
    assert concept_corpus.token2id == {'a/b': 0, 'd/c': 1, '*/e': 2, 'c/e': 3, 'd/e': 4, '*/d': 5}

    #  1    6    7    8    9   10
    # a/b  c/d  */e  d/e  c/e  */d

    assert (
        (
            concept_corpus.data.todense()
            == np.matrix(
                [
                    # 0  1  2  3  4  5
                    [0, 5, 0, 1, 1, 0],
                    [0, 3, 0, 1, 1, 2],
                    [0, 0, 1, 0, 3, 1],
                ],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )

    assert concept_corpus.term_frequency.tolist() == [0, 8, 1, 2, 5, 3]
    assert concept_corpus.overridden_term_frequency is None

    """ Slice full corpus """
    corpus = bundle.corpus

    # pp(corpus.data.todense())
    assert (
        (
            corpus.data.todense()
            == np.matrix(
                [
                    # 0  1  2  3  4  5  6  7  8  9 10 11 12
                    [1, 2, 1, 1, 5, 3, 1, 1, 2, 1, 0, 0, 0],
                    [2, 0, 0, 0, 3, 0, 3, 0, 4, 1, 2, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 2, 0, 3, 1, 0, 3],
                ],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )

    corpus.slice_by_indices(keep_ids, inplace=True)

    # pp(corpus.data.todense())
    assert (
        (
            corpus.data.todense()
            == np.matrix(
                [
                    # 1  6  7  8  9 10
                    # ----------------
                    # 0  1  2  3  4  5
                    # ----------------
                    [2, 5, 1, 2, 1, 0],
                    [0, 3, 0, 4, 1, 2],
                    [0, 0, 2, 0, 3, 1],
                    # ----------------
                    # 2  8  3  5  6  3
                ],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )

    assert corpus.token2id == {'a/b': 0, 'd/c': 1, '*/e': 2, 'c/e': 3, 'd/e': 4, '*/d': 5}
    assert corpus.term_frequency.tolist() == [2, 8, 3, 6, 5, 3]
    assert corpus.overridden_term_frequency is None

    """Update token count and token2id"""

    def _token_ids_to_keep(kept_pair_ids: Set[int]) -> List[int]:
        """Returns sorted token IDs that given co-occurrence pair IDs corresponds to """
        token_ids_in_kept_pairs: Set[int] = set(
            flatten((k for k, pair_id in bundle.token_ids_2_pair_id.items() if pair_id in kept_pair_ids))
        )
        kept_token_ids: List[int] = sorted(list(token_ids_in_kept_pairs.union(set(bundle.token2id.magic_token_ids))))
        return kept_token_ids

    """" Inlined calls """
    token_ids_in_kept_pairs: Set[int] = set(
        flatten((k for k, pair_id in bundle.token_ids_2_pair_id.items() if pair_id in keep_ids))
    )
    assert token_ids_in_kept_pairs == {0, 1, 3, 4, 5, 6}  # all except masked token

    kept_token_ids = sorted(list(token_ids_in_kept_pairs.union(set(bundle.token2id.magic_token_ids))))
    assert kept_token_ids == [0, 1, 2, 3, 4, 5, 6]

    """" Equals function call """
    assert kept_token_ids == _token_ids_to_keep(set(keep_ids))

    assert (
        (
            corpus.window_counts.document_term_window_counts.todense()
            == np.matrix(
                [
                    # *  -  a  b  c  d  e
                    [2, 3, 0, 2, 3, 6, 2],
                    [2, 3, 0, 3, 0, 5, 3],
                    [2, 2, 0, 0, 2, 0, 4],
                ],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )

    corpus.window_counts.slice(kept_token_ids, inplace=True)

    """Nothing is changed since all original tokens are kepts"""
    assert corpus.window_counts.document_term_window_counts.shape == (3, 7)

    """Simulate removed token `b` """
    wc: TokenWindowCountMatrix = corpus.window_counts.slice([x for x in kept_token_ids if x != 3], inplace=False)

    assert (
        (
            wc.document_term_window_counts.todense()
            == np.matrix(
                [[2, 3, 0, 3, 6, 2], [2, 3, 0, 0, 5, 3], [2, 2, 0, 2, 0, 4]],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )

    wc: TokenWindowCountMatrix = corpus.window_counts

    wc: TokenWindowCountMatrix = concept_corpus.window_counts
    assert (
        (
            wc.slice(kept_token_ids, inplace=False).document_term_window_counts.todense()
            == np.matrix(
                [[0, 3, 0, 0, 0, 3, 1], [1, 3, 0, 0, 0, 2, 1], [1, 2, 0, 0, 0, 0, 2]],
                dtype=np.int32,
            )
        )
        .all()
        .all()
    )
    assert ids_translation == {1: 0, 4: 1, 7: 2, 8: 3, 9: 4, 10: 5}
    translated_token2id = bundle.token2id.translate(ids_translation, inplace=False)
    assert translated_token2id is not None

    bundle._token_ids_2_pair_id = {
        pair: pair_id for pair, pair_id in bundle._token_ids_2_pair_id if pair_id in ids_translation
    }
