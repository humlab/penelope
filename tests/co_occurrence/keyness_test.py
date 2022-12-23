# type: ignore
# pylint: disable=unused-import

import collections
from itertools import combinations
from pprint import pprint as pp
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pytest
import scipy

from penelope.co_occurrence import Bundle, ContextOpts, VectorizedTTM, VectorizeType, generate_windows, windows_to_ttm
from penelope.co_occurrence.keyness import ComputeKeynessOpts, compute_corpus_keyness, significance_ratio
from penelope.common.keyness import KeynessMetric, KeynessMetricSource, metrics
from penelope.corpus import Token2Id, VectorizedCorpus
from penelope.corpus.tokenized_corpus import TokenizedCorpus
from penelope.pipeline.co_occurrence.tasks_pool import tokens_to_ttm
from penelope.utility import faster_to_dict_records, flatten
from tests.utils import inline_code

from .utils import create_keyness_opts, create_keyness_test_bundle, create_simple_bundle_by_pipeline, very_simple_corpus

SIMPLE_CORPUS_ABCDE_3DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c', 'd', 'c', 'e']),
    ('tran_2019_02_test.txt', ['a', 'a', 'c', 'e', 'c', 'd', 'd']),
    ('tran_2019_03_test.txt', ['d', 'e', 'e', 'b']),
]

# pylint: disable=protected-access


def test_tasks_pool_tokens_to_ttm_step_by_step():

    # Arrange
    context_opts: ContextOpts = ContextOpts(
        concept={'d'},
        ignore_concept=False,
        ignore_padding=False,
        context_width=1,
        processes=None,
    )
    corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_3DOCS)

    pad_id = len(corpus.token2id)
    corpus.token2id[context_opts.pad] = pad_id

    token2id: dict = corpus.token2id
    id2token: dict = corpus.id2token

    filename, tokens = next(corpus)
    # doc_info = corpus.document_index[corpus.document_index.filename == filename].to_dict('records')[0]
    doc_info = faster_to_dict_records(corpus.document_index[corpus.document_index.filename == filename])[0]
    document_id = doc_info['document_id']
    token_ids = [token2id[t] for t in tokens]
    concept_ids = {token2id[x] for x in context_opts.concept}

    # Act

    windows: Iterable[Iterable[int]] = generate_windows(
        token_ids=token_ids,
        context_width=context_opts.context_width,
        pad_id=pad_id,
        ignore_pads=context_opts.ignore_padding,
    )

    # Assert
    windows = [w for w in windows]
    assert windows == [[5, 0, 1], [0, 1, 2], [1, 2, 2], [2, 2, 3], [2, 3, 2], [3, 2, 4], [2, 4, 5]]
    assert [[id2token[i] for i in w] for w in windows] == [
        ['*', 'a', 'b'],
        ['a', 'b', 'c'],
        ['b', 'c', 'c'],
        ['c', 'c', 'd'],
        ['c', 'd', 'c'],
        ['d', 'c', 'e'],
        ['c', 'e', '*'],
    ]
    # ['a', 'b', 'c', 'c', 'd', 'c', 'e']
    ttm_map: Mapping[VectorizeType, VectorizedTTM] = windows_to_ttm(
        document_id=document_id,
        windows=windows,
        concept_ids=concept_ids,
        ignore_ids=set(),
        vocab_size=len(token2id),
    )
    expected_normal_ttm = [
        # a  b  c  d  e  *
        [0, 2, 1, 0, 0, 1],  # a
        [0, 0, 3, 0, 0, 1],  # b
        [0, 0, 0, 5, 2, 1],  # c
        [0, 0, 0, 0, 1, 0],  # d
        [0, 0, 0, 0, 0, 1],  # e
        [0, 0, 0, 0, 0, 0],  # *
    ]
    assert (ttm_map[VectorizeType.Normal].term_term_matrix.todense() == expected_normal_ttm).all()

    expected_concept_ttm = [
        # a  b  c  d  e  *
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 5, 1, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]

    assert (ttm_map[VectorizeType.Concept].term_term_matrix.todense() == expected_concept_ttm).all()


def test_compute_ttm_alternative_method():

    corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_3DOCS)
    corpus.token2id['*'] = (pad_id := len(corpus.token2id))
    id2token: dict = corpus.id2token

    """Convert corpus to numeric ids """
    corpus_token_ids = [[corpus.token2id[t] for t in tokens] for _, tokens in corpus]

    corpus_document_windows = [
        [
            w
            for w in generate_windows(
                token_ids=token_ids,
                context_width=1,
                pad_id=pad_id,
                ignore_pads=False,
            )
        ]
        for token_ids in corpus_token_ids
    ]

    corpus_document_text_windows = [[''.join([id2token[t] for t in w]) for w in d] for d in corpus_document_windows]
    corpus_document_text_windows = flatten(corpus_document_text_windows)

    assert corpus_document_text_windows == flatten(
        [
            ['*ab', 'abc', 'bcc', 'ccd', 'cdc', 'dce', 'ce*'],
            ['*aa', 'aac', 'ace', 'cec', 'ecd', 'cdd', 'dd*'],
            ['*de', 'dee', 'eeb', 'eb*'],
        ]
    )

    co_occurrence_instances = flatten(list(map(''.join, combinations(x, 2))) for x in corpus_document_text_windows)
    co_occurrence_counts = collections.Counter(
        x if x[0] < x[1] else x[::-1] for x in co_occurrence_instances if x[0] != x[1]
    )

    assert dict(co_occurrence_counts) == {
        '*a': 3,
        '*b': 2,
        'ab': 2,
        'ac': 4,
        'bc': 3,
        'cd': 8,
        'de': 5,
        'ce': 6,
        '*c': 1,
        '*e': 3,
        'ae': 1,
        '*d': 3,
        'be': 3,
    }

    assert True


def test_tasks_pool_tokens_to_ttm():
    corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_3DOCS)
    token2id: dict = corpus.token2id
    context_opts: ContextOpts = ContextOpts(
        concept={'d'},
        ignore_concept=False,
        ignore_padding=False,
        context_width=1,
        processes=None,
        windows_threshold=0,
    )
    token2id[context_opts.pad] = len(token2id)
    concept_ids = {token2id[x] for x in context_opts.concept}
    ignore_ids = set()
    filename, tokens = next(corpus)
    # doc_info = corpus.document_index[corpus.document_index.filename == filename].to_dict('records')[0]
    doc_info = faster_to_dict_records(corpus.document_index[corpus.document_index.filename == filename])[0]
    token_ids = [token2id[t] for t in tokens]
    pad_id = token2id[context_opts.pad]
    args = (
        doc_info['document_id'],
        doc_info['document_name'],
        doc_info['filename'],
        token_ids,
        pad_id,
        context_opts,
        concept_ids,
        ignore_ids,
        len(token2id),
    )

    item: dict = tokens_to_ttm(args)
    assert item is not None


def test_keyness_transform_with_simple_corpus():

    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS)
    opts: ComputeKeynessOpts = create_keyness_opts()

    corpus: VectorizedCorpus = bundle.keyness_transform(opts=opts)

    assert corpus is not None


def test_step_by_step_tfidf_keyness_transform():

    bundle: Bundle = create_keyness_test_bundle(data=SIMPLE_CORPUS_ABCDE_3DOCS, processes=None, ignore_padding=False)
    opts: ComputeKeynessOpts = create_keyness_opts()

    corpus: VectorizedCorpus = bundle.corpus
    concept_corpus: VectorizedCorpus = bundle.concept_corpus

    """ STEP: Reduce corpus size if TF threshold is specified
        @filename: keyness.py, compute_weighed_corpus_keyness:75"""

    """Expected result (see test_compute_ttm_alternative_method above)"""
    expected_result = {
        '*/a': 3,
        'a/b': 2,
        '*/b': 2,
        '*/c': 1,
        'b/c': 3,
        'a/c': 4,
        'c/d': 8,
        '*/e': 3,
        'd/e': 5,
        'c/e': 6,
        '*/d': 3,
        'a/e': 1,
        'b/e': 3,
    }
    fg = corpus.id2token.get
    tf = {fg(i): x for i, x in enumerate(corpus.term_frequency)}
    # tf = {x if x[0] < x[2] else x[::-1]: i for x,i in tf.items()}

    assert tf == expected_result

    zero_out_indices: Sequence[int] = corpus.zero_out_by_tf_threshold(3)
    assert zero_out_indices.tolist() == [1, 2, 3, 11]
    assert corpus.term_frequency.tolist() == [3, 0, 0, 0, 3, 4, 8, 3, 5, 6, 3, 0, 3]

    assert concept_corpus.term_frequency.tolist() == [0, 0, 0, 0, 0, 0, 8, 1, 5, 2, 3, 0, 0]
    concept_corpus.zero_out_by_indices(zero_out_indices)
    assert concept_corpus.term_frequency.tolist() == [0, 0, 0, 0, 0, 0, 8, 1, 5, 2, 3, 0, 0]

    """ STEP: Compute corpus keyness for both corpora
        @filename: penelope/co_occurrence/keyness.py:23, compute_corpus_keyness
        Compute keyness (TF-IDF in this case - must be done before grouping)
    """
    with inline_code(source=compute_corpus_keyness):
        assert (
            (
                corpus.data.todense()
                == np.matrix(
                    [
                        [1, 0, 0, 0, 3, 1, 5, 1, 1, 2, 0, 0, 0],
                        [2, 0, 0, 0, 0, 3, 3, 0, 1, 4, 2, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 1, 0, 3],
                    ],
                    dtype=np.int32,
                )
            )
            .all()
            .all()
        )

        corpus = corpus.tf_idf()
        corpus = corpus.group_by_time_period_optimized(
            time_period_specifier=opts.temporal_pivot,
            target_column_name=opts.pivot_column_name,
            fill_gaps=opts.fill_gaps,
        )

        concept_corpus = concept_corpus.tf_idf()
        concept_corpus = concept_corpus.group_by_time_period_optimized(
            time_period_specifier=opts.temporal_pivot,
            target_column_name=opts.pivot_column_name,
            fill_gaps=opts.fill_gaps,
        )

        M: scipy.sparse.spmatrix = significance_ratio(concept_corpus.data, corpus.data)

    assert M is not None


def test_significant_ratio():
    A = scipy.sparse.csr_matrix(np.array([[1, 2, 4], [2, 4, 5], [25, 15, 20]]))
    B = A.copy()

    R = metrics.significance_ratio(A, B)

    assert (R == 1.0).todense().all().all()

    B = scipy.sparse.csr_matrix(np.array([[1, 2, 2], [4, 1, 25], [5, 5, 0]]))
    R = metrics.significance_ratio(A, B)
    assert (
        (R.todense() == scipy.sparse.csr_matrix(np.array([[1.0, 1.0, 2.0], [0.5, 4.0, 0.2], [5.0, 3.0, 0.0]])))
        .all()
        .all()
    )


@pytest.mark.parametrize(
    "tag,keyness_source,keyness",
    [
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR_Z),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.LLR_N),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.TF_normalized),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR_Z),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.LLR_N),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Concept, KeynessMetric.TF_normalized),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.PPMI),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.HAL_cwr),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR_Z),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.LLR_N),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.DICE),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.TF_IDF),
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Weighed, KeynessMetric.TF_normalized),
    ],
)
def test_keyness_transform_corpus(tag: str, keyness_source: KeynessMetricSource, keyness: KeynessMetric):
    folder: str = f'./tests/test_data/{tag}'
    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)
    opts: ComputeKeynessOpts = ComputeKeynessOpts(
        temporal_pivot="year",
        keyness_source=keyness_source,
        keyness=keyness,
        tf_threshold=1,
        pivot_column_name='time_period',
        normalize=False,
        fill_gaps=False,
    )
    corpus: VectorizedCorpus = bundle.keyness_transform(opts=opts)

    assert corpus is not None


@pytest.mark.parametrize(
    "tag,keyness_source,keyness",
    [
        ('ABCDEFG_7DOCS_CONCEPT', KeynessMetricSource.Full, KeynessMetric.HAL_cwr),
    ],
)
def test_keyness_transform_corpus2(tag: str, keyness_source: KeynessMetricSource, keyness: KeynessMetric):
    folder: str = f'./tests/test_data/{tag}'
    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)
    opts: ComputeKeynessOpts = ComputeKeynessOpts(
        temporal_pivot="year",
        keyness_source=keyness_source,
        keyness=keyness,
        tf_threshold=10,
        pivot_column_name='time_period',
        normalize=False,
        fill_gaps=False,
    )
    corpus: VectorizedCorpus = bundle.keyness_transform(opts=opts)

    assert corpus is not None


def test_zero_out_by_tf_threshold():
    expected_sums = [28, 12, 9, 11, 39, 34, 7, 8, 15, 16, 10, 34, 8, 28, 14, 19, 28, 23, 23, 9, 16, 9, 16, 4, 16, 17, 4]
    tag: str = 'ABCDEFG_7DOCS_CONCEPT'
    folder: str = f'./tests/test_data/{tag}'
    bundle: Bundle = Bundle.load(folder=folder, tag=tag, compute_frame=False)

    corpus: VectorizedCorpus = bundle.corpus

    assert (corpus.term_frequency == expected_sums).all()

    tf_threshold: int = 10
    indices = [i for i, v in enumerate(expected_sums) if v < tf_threshold]
    for i in indices:
        expected_sums[i] = 0

    corpus.zero_out_by_tf_threshold(tf_threshold)

    assert (corpus.term_frequency == expected_sums).all()
