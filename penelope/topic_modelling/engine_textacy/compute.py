import types
from typing import Any, Dict, Iterable, Union

import gensim
import scipy
import textacy

import penelope.vendor.textacy as textacy_utility


def compute(
    doc_term_matrix: scipy.sparse.csr_matrix,
    terms: Iterable[Iterable[str]],
    id2word: Union[gensim.corpora.Dictionary, Dict[int, str]],
    vectorizer_args: Dict[str, Any],
    method: str,
    engine_args: Dict[str, Any],
    tfidf_weiging: bool = False,  # pylint: disable=unused-argument
):
    """Computes a topic model using Gensim as engine.

    Parameters
    ----------
    doc_term_matrix : scipy.sparse.csr_matrix
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
        model               The textaCy topic model
        doc_topic_matrix    (None)
        vectorizer_args     Used vectorizer args (if any)
        perplexity_score    Computed perplexity scores
        coherence_score     Computed coherence scores
        engine_options      Used engine options (algorithm specific)
    """
    if doc_term_matrix is None:
        assert terms is not None
        doc_term_matrix, id2word = textacy_utility.vectorize_terms(terms, vectorizer_args)

    model = textacy.tm.TopicModel(method.split('_')[1], **engine_args)
    model.fit(doc_term_matrix)

    doc_topic_matrix = model.transform(doc_term_matrix)

    # We use gensim's corpus as common result format
    corpus = gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)

    return types.SimpleNamespace(
        corpus=corpus,
        doc_term_matrix=doc_term_matrix,
        id2word=id2word,
        model=model,
        doc_topic_matrix=doc_topic_matrix,
        vectorizer_args=vectorizer_args,
        perplexity_score=None,
        coherence_score=None,
        engine_options=engine_args,
    )
