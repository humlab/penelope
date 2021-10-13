from __future__ import annotations

from typing import Any, Dict

from ..interfaces import InferredModel, TrainingCorpus
from . import coherence, options


def train(
    train_corpus: TrainingCorpus,
    method: str,
    engine_args: Dict[str, Any],
    **kwargs: Dict[str, Any],
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
        options:
            perplexity_score    Computed perplexity scores
            coherence_score     Computed coherence scores
            engine_ptions       Passed engine options (not the interpreted algorithm specific options)
            extra_options       Any other compute option passed as a kwarg
    """
    algorithm_name: str = method.split('_')[1].upper()

    train_corpus.to_sparse_corpus()

    if kwargs.get('tfidf_weiging', False):
        train_corpus.to_tf_idf()

    algorithm: dict = options.get_engine_options(
        algorithm=algorithm_name,
        corpus=train_corpus.corpus,
        id2word=train_corpus.id2token,
        engine_args=engine_args,
    )

    engine = algorithm['engine']
    engine_options = algorithm['options']

    model = engine(**engine_options)

    # FIXME: These metrics must be computed on a held-out corpus - not the training corpus
    perplexity_score = (
        None
        if not hasattr(model, 'log_perplexity')
        else 2 ** model.log_perplexity(train_corpus.corpus, len(train_corpus.corpus))
    )

    coherence_score = coherence.compute_score(train_corpus.id2token, model, train_corpus.corpus)

    return InferredModel(
        train_corpus=train_corpus,
        topic_model=model,
        method=method,
        perplexity_score=perplexity_score,
        coherence_score=coherence_score,
        engine_options=engine_args,
        extra_options=kwargs,
    )
