from penelope.co_occurrence import Bundle, ContextOpts, CoOccurrenceComputeResult
from penelope.co_occurrence.partition_by_document import (
    co_occurrences_to_vectorized_corpus,
    compute_corpus_co_occurrence,
)
from penelope.corpus import ITokenizedCorpus, Token2Id


def create_co_occurrence_bundle(
    *, corpus: ITokenizedCorpus, context_opts: ContextOpts, folder: str, tag: str
) -> Bundle:

    token2id: Token2Id = Token2Id(corpus.token2id)

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        document_index=corpus.document_index,
        token2id=token2id,
        context_opts=context_opts,
        global_threshold_count=1,
    )

    corpus = co_occurrences_to_vectorized_corpus(
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        token2id=token2id,
    )

    bundle: Bundle = Bundle(
        folder=folder,
        tag=tag,
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        token_window_counts=value.token_window_counts,
        token2id=value.token2id,
        compute_options={},
        corpus=corpus,
    )

    return bundle
