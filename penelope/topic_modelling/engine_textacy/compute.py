from typing import Any, Dict

import gensim
import penelope.vendor.textacy as textacy_utility
import textacy

from ..container import InferredModel, TrainingCorpus


def compute(
    train_corpus: TrainingCorpus,
    method: str,
    engine_args: Dict[str, Any],
    **kwargs,  # pylint: disable=unused-argument
) -> InferredModel:
    """Computes a topic model using Gensim as engine.

    Parameters
    ----------
    train_corpus : TrainingCorpus
        A container for the training corpus data (terms or DTM, id2word, document_index)
    method : str
        The method to use (see `options` module for mappings)
    engine_args : Dict[str, Any]
        Generic topic modelling options that are translated to algorithm-specific options (see `options` module for translation)
    kwargs : Dict[str,Any], optional
        Additional options:
            `tfidf_weiging` if TF-IDF weiging should be applied, ony valid when terms/id2word are specified, by default False

    Returns
    -------
    InferredModel
        train_corpus        Training corpus data (updated)
        model               The textaCy topic model
        perplexity_score    Computed perplexity scores
        coherence_score     Computed coherence scores
        engine_ptions       Used engine options (algorithm specific)
        extra_options       Any other compute option passed as a kwarg
    """
    if train_corpus.doc_term_matrix is None:
        assert train_corpus.terms is not None
        train_corpus.doc_term_matrix, train_corpus.id2word = textacy_utility.vectorize_terms(
            train_corpus.terms, train_corpus.vectorizer_args
        )

    model = textacy.tm.TopicModel(method.split('_')[1], **engine_args)

    model.fit(train_corpus.doc_term_matrix)

    # We use gensim's corpus as common result format
    train_corpus.corpus = gensim.matutils.Sparse2Corpus(train_corpus.doc_term_matrix, documents_columns=False)

    return InferredModel(
        train_corpus=train_corpus,
        topic_model=model,
        method=method,
        perplexity_score=None,
        coherence_score=None,
        engine_options=engine_args,
        extra_options=kwargs,
    )
