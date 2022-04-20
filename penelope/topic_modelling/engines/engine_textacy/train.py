from typing import Any, Dict

from penelope import corpus as pc
from penelope.corpus.dtm import convert
from penelope.utility import deprecated
from penelope.vendor import textacy_api

from ...interfaces import InferredModel, TrainingCorpus


@deprecated
def train(
    train_corpus: TrainingCorpus,
    method: str,
    engine_args: Dict[str, Any],
    **kwargs,
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
            `tfidf_weighing` if TF-IDF weighing should be applied, ony valid when terms/id2word are specified, by default False

    Returns
    -------
    InferredModel
        train_corpus        Training corpus data (updated)
        model               The textaCy topic model
        perplexity_score    Computed perplexity scores
        coherence_score     Computed coherence scores
        engine_options       Used engine options (algorithm specific)
        extra_options       Any other compute option passed as a kwarg
    """

    corpus: pc.VectorizedCorpus = convert.TranslateCorpus.translate(
        train_corpus.corpus,
        token2id=train_corpus.token2id.data,
        document_index=train_corpus.document_index,
        vectorize_opts=pc.VectorizeOpts().update(**kwargs),
    )

    model = textacy_api.TopicModel(method.split('_')[1], **engine_args)

    model.fit(corpus.data)

    train_corpus.corpus = corpus

    return InferredModel(
        topic_model=model,
        id2token=train_corpus.id2token,
        options=dict(
            method=method,
            perplexity_score=None,
            coherence_score=None,
            engine_options=engine_args,
            extra_options=kwargs,
        ),
    )
