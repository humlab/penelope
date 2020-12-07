from typing import Any, Dict

import gensim

from ..container import InferredModel, TrainingCorpus
from . import coherence, options


def compute(
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
    algorithm_name = method.split('_')[1].upper()

    if train_corpus.doc_term_matrix is None:
        train_corpus.id2word = gensim.corpora.Dictionary(train_corpus.terms)
        bow_corpus = [train_corpus.id2word.doc2bow(tokens) for tokens in train_corpus.terms]
        csc_matrix = gensim.matutils.corpus2csc(
            bow_corpus, num_terms=len(train_corpus.id2word), num_docs=len(bow_corpus), num_nnz=sum(map(len, bow_corpus))
        )
        train_corpus.corpus = gensim.matutils.Sparse2Corpus(csc_matrix, documents_columns=True)
    else:
        assert train_corpus.id2word is not None
        train_corpus.corpus = gensim.matutils.Sparse2Corpus(train_corpus.doc_term_matrix, documents_columns=False)

    if kwargs.get('tfidf_weiging', False):
        train_corpus.corpus = _tfi_idf_model(train_corpus.corpus)

    algorithm = options.engine_options(algorithm_name, train_corpus.corpus, train_corpus.id2word, engine_args)

    engine = algorithm['engine']
    engine_options = algorithm['options']

    model = engine(**engine_options)

    # FIXME: These metrics must be computed on a held-out corpus - not the training corpus
    perplexity_score = (
        None
        if not hasattr(model, 'log_perplexity')
        else 2 ** model.log_perplexity(train_corpus.corpus, len(train_corpus.corpus))
    )

    coherence_score = coherence.compute_score(train_corpus.id2word, model, train_corpus.corpus)

    return InferredModel(
        train_corpus=train_corpus,
        topic_model=model,
        method=method,
        perplexity_score=perplexity_score,
        coherence_score=coherence_score,
        engine_options=engine_args,
        extra_options=kwargs,
    )


def _tfi_idf_model(corpus):
    # assert algorithm_name != 'MALLETLDA', 'MALLET training model cannot (currently) use TFIDF weighed corpus'
    tfidf_model = gensim.models.tfidfmodel.TfidfModel(corpus)
    corpus = [tfidf_model[d] for d in corpus]
    return corpus
