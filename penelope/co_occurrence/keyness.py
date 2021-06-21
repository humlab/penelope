from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy
from penelope.common.keyness import KeynessMetric, KeynessMetricSource, significance_ratio

from ..corpus import Token2Id, VectorizedCorpus


@dataclass
class ComputeKeynessOpts:

    keyness_source: KeynessMetricSource
    keyness: KeynessMetric
    period_pivot: str
    pivot_column_name: str
    normalize: bool = False
    fill_gaps: bool = False
    tf_threshold: Union[int, float] = 1


def compute_corpus_keyness(corpus: VectorizedCorpus, opts: ComputeKeynessOpts, token2id: Token2Id) -> VectorizedCorpus:

    """Metrics computed on a document level"""
    if opts.keyness == KeynessMetric.TF_IDF:
        corpus = corpus.tf_idf()
    elif opts.keyness == KeynessMetric.TF_normalized:
        corpus = corpus.normalize_by_raw_counts()
    elif opts.keyness == KeynessMetric.TF:
        pass

    corpus = corpus.group_by_time_period_optimized(
        time_period_specifier=opts.period_pivot,
        target_column_name=opts.pivot_column_name,
        fill_gaps=opts.fill_gaps,
    )

    """Metrics computed on partitioned corpus"""
    if opts.keyness in (KeynessMetric.PPMI, KeynessMetric.LLR, KeynessMetric.DICE, KeynessMetric.LLR_Dunning):
        corpus = corpus.to_keyness_co_occurrence_corpus(
            keyness=opts.keyness,
            token2id=token2id,
            pivot_key=opts.pivot_column_name,
            normalize=opts.normalize,
        )
    elif opts.keyness == KeynessMetric.HAL_cwr:
        corpus = corpus.HAL_cwr_corpus()

    return corpus


def compute_weighed_corpus_keyness(
    corpus: VectorizedCorpus,
    concept_corpus: Optional[VectorizedCorpus],
    single_vocabulary: Token2Id,
    opts: ComputeKeynessOpts,
) -> VectorizedCorpus:
    """Computes a keyness corpus for `corpus` and optionally `concept_corpus`.

    Args:
        corpus (VectorizedCorpus): [description]
        concept_corpus (VectorizedCorpus): [description]
        single_vocabulary (Token2Id): [description]
        vocabs_mapping (VocabularyMapping): Mapping between single/pair vocabs
        opts (KeynessOpts): Compute opts

    Raises:
        ValueError: [description]

    Returns:
        VectorizedCorpus: [description]
    """

    if opts.period_pivot not in ["year", "lustrum", "decade"]:
        raise ValueError(f"illegal time period {opts.period_pivot}")

    if opts.keyness_source in (KeynessMetricSource.Concept, KeynessMetricSource.Weighed):
        if concept_corpus is None:
            raise ValueError(f"Keyness {opts.keyness_source.name} requested when concept corpus is None!")

    if opts.tf_threshold > 1:
        # corpus = corpus.slice_by_term_frequency(opts.tf_threshold)
        indicies = np.argwhere(corpus.term_frequencies < opts.tf_threshold).ravel()
        if len(indicies) > 0:
            corpus.data[:, indicies] = 0
            corpus.data.eliminate_zeros()
            if concept_corpus is not None:
                concept_corpus.data[:, indicies] = 0
                concept_corpus.data.eliminate_zeros()

    corpus: VectorizedCorpus = (
        compute_corpus_keyness(corpus=corpus, opts=opts, token2id=single_vocabulary)
        if opts.keyness_source in (KeynessMetricSource.Full, KeynessMetricSource.Weighed)
        else None
    )

    concept_corpus: VectorizedCorpus = (
        compute_corpus_keyness(corpus=concept_corpus, opts=opts, token2id=single_vocabulary)
        if opts.keyness_source in (KeynessMetricSource.Concept, KeynessMetricSource.Weighed)
        else None
    )

    weighed_corpus: VectorizedCorpus = (
        weigh_corpora(corpus, concept_corpus) if corpus and concept_corpus else corpus or concept_corpus
    )

    return weighed_corpus


def weigh_corpora(corpus: VectorizedCorpus, concept_corpus: VectorizedCorpus) -> VectorizedCorpus:

    if corpus.data.shape != concept_corpus.data.shape:
        raise ValueError("Corpus shapes doesn't match")

    M: scipy.sparse.spmatrix = significance_ratio(concept_corpus.data, corpus.data)

    weighed_corpus: VectorizedCorpus = VectorizedCorpus(
        bag_term_matrix=M,
        token2id=concept_corpus.token2id,
        document_index=concept_corpus.document_index,
        term_frequency_mapping=concept_corpus.term_frequency_mapping,
        **concept_corpus.payload,
    )
    return weighed_corpus
