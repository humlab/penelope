from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from ...interfaces import InferredModel, TrainingCorpus
from . import coherence, convert, options


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
        A container for the training data (terms or DTM, id2word, document_index)
    method : str
        The method to use (see `options` module for mappings)
    engine_args : Dict[str, Any]
        Generic topic modelling options that are translated to algorithm-specific options (see `options` module for translation)
    kwargs : Dict[str,Any], optional
        Additional vectorize options:
            `tfidf_weiging` if TF-IDF weiging should be applied, ony valid when terms/id2word are specified, by default False

    Returns
    -------
    InferredModel
        train_corpus        Training corpus data (updated)
        model               The engine specific topic model
        options:
            perplexity_score    Computed perplexity scores
            coherence_score     Computed coherence scores
            engine_ptions       Passed engine options (not the interpreted algorithm specific options)
            extra_options       Any other compute option passed as a kwarg
    """

    corpus, dictionary = convert.TranslateCorpus().translate(train_corpus.corpus, id2token=train_corpus.id2token)

    if kwargs.get('tfidf_weiging', False):
        logger.warning("TF-IDF weighing of effective corpus has been disabled")
        # tfidf_model = TfidfModel(corpus)
        # corpus = [tfidf_model[d] for d in corpus]

    # todo: translate to VectorizedCorpus?
    train_corpus.effective_corpus = corpus
    if train_corpus.token2id is None:
        train_corpus.token2id = dictionary.token2id

    engine_spec: options.EngineSpec = options.get_engine_specification(engine_key=method)
    model = engine_spec.engine(
        **engine_spec.get_options(
            corpus=train_corpus.effective_corpus,
            id2word=train_corpus.id2token,
            engine_args=engine_args,
        )
    )

    # FIXME: These metrics must be computed on a held-out corpus - not the training corpus
    perplexity_score = (
        None
        if not hasattr(model, 'log_perplexity')
        else 2 ** model.log_perplexity(train_corpus.effective_corpus, len(train_corpus.effective_corpus))
    )

    coherence_score = coherence.compute_score(train_corpus.id2token, model, train_corpus.effective_corpus)

    return InferredModel(
        train_corpus=train_corpus,
        topic_model=model,
        method=method,
        perplexity_score=perplexity_score,
        coherence_score=coherence_score,
        engine_options=engine_args,
        extra_options=kwargs,
    )
