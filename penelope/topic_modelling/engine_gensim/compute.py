import types
from typing import Any, Dict, Iterable, Union

import gensim
import scipy
import penelope.topic_modelling.engine_gensim.coherence as coherence

from . import options


def compute(
    doc_term_matrix: scipy.sparse.csr_matrix,
    terms: Iterable[Iterable[str]],
    id2word: Union[gensim.corpora.Dictionary, Dict[int, str]],
    vectorizer_args: Dict[str, Any],
    method: str,
    engine_args: Dict[str, Any],
    tfidf_weiging: bool = False,
):
    """Computes a topic model using Gensim as engine.

    Parameters
    ----------
    doc_term_matrix : scipy.sparse,csr_matrix
        A DTM matrix, optional
    terms : Iterable[Iterable[str]]
        A document token stream, mandatory if `doc_term_matrix` is None, otherwise optional
    id2word : Union[gensim.corpora.Dictionary, Dict[int, str]]
        A dictionary i.e. id-to-word mapping, mandatory if `doc_term_matrix` is not None, otherwise created
    vectorizer_args : Dict[str, Any]
        Arguments to use if vectorizing is needed i.e. if `doc_term_matrix` is None
    method : str
        The method to use (see `options` module for mappings)
    engine_args : Dict[str, Any]
        Generic topic modelling options that are translated to algorithm-specific options (see `options` module for translation)
    tfidf_weiging : bool, optional
        Flag if TF-IDF weiging should be applied, ony valid when terms/id2word are specified, by default False

    Returns
    -------
    types.SimpleNamespace
        corpus              Gensim corpus,
        doc_term_matrix     The DTM
        id2word             The Gensim Dictionary
        model               The Gensim topic Model
        doc_topic_matrix    (None)
        vectorizer_args     Used vectorizer args (if any)
        perplexity_score    Computed perplexity scores
        coherence_score     Computed coherence scores
        engine_options      Used engine options (algorithm specific)
    """
    algorithm_name = method.split('_')[1].upper()

    if doc_term_matrix is None:
        id2word = gensim.corpora.Dictionary(terms)
        corpus = [id2word.doc2bow(tokens) for tokens in terms]
    else:
        assert id2word is not None
        corpus = gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)

    if tfidf_weiging:
        # assert algorithm_name != 'MALLETLDA', 'MALLET training model cannot (currently) use TFIDF weighed corpus'
        tfidf_model = gensim.models.tfidfmodel.TfidfModel(corpus)
        corpus = [tfidf_model[d] for d in corpus]

    algorithm = options.engine_options(algorithm_name, corpus, id2word, engine_args)

    engine = algorithm['engine']
    engine_options = algorithm['options']

    model = engine(**engine_options)

    perplexity_score = None if not hasattr(model, 'log_perplexity') else 2 ** model.log_perplexity(corpus, len(corpus))

    coherence_score = coherence.compute_score(id2word, model, corpus)

    return types.SimpleNamespace(
        corpus=corpus,
        doc_term_matrix=doc_term_matrix,
        id2word=id2word,
        model=model,
        doc_topic_matrix=None,
        vectorizer_args=vectorizer_args,
        perplexity_score=perplexity_score,
        coherence_score=coherence_score,
        engine_options=engine_options,
    )
