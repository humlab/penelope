from typing import Mapping

import pandas as pd
import scipy

from penelope.corpus import tokenized_corpus
from penelope.corpus import vectorizer as corpus_vectorizer
from penelope.corpus.tokenized_corpus import TokenizedCorpus


def reader_coocurrence_matrix(reader, min_count: int = 1, **kwargs) -> pd.DataFrame:
    """Computes a term-term coocurrence matrix for documents in reader.

    Parameters
    ----------
    reader : Iterable[List[str]]
        Sequence of tokenized documents

    Returns
    -------
    pd.DataFrame
        Upper diagonal of term-term frequency matrix (TTM). Note that diagonal (wi, wi) is not returned
    """
    corpus = tokenized_corpus.TokenizedCorpus(reader, only_alphanumeric=False, **kwargs)

    vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
    v_corpus = vectorizer.fit_transform(corpus)
    term_term_matrix = v_corpus.cooccurrence_matrix()

    coo_df = cooccurrence_matrix_to_dataframe(term_term_matrix, v_corpus.id2token, corpus.documents, min_count)

    return coo_df


def corpus_to_coocurrence_matrix(corpus: TokenizedCorpus) -> scipy.sparse.spmatrix:
    """Computes a term-term coocurrence matrix for documents in reader.

    Parameters
    ----------
    corpus : TokenizedCorpus
        Sequence of tokenized documents

    Returns
    -------
    scipy.sparse.spmatrix
        Upper diagonal of term-term frequency matrix (TTM). Note that diagonal (wi, wi) is not returned
    """

    vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
    v_corpus = vectorizer.fit_transform(corpus)
    term_term_matrix = v_corpus.cooccurrence_matrix()

    return term_term_matrix


def cooccurrence_matrix_to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    id2token: Mapping[int, str],
    documents: pd.DataFrame = None,
    min_count: int = 1,
):
    """Converts a TTM to a Pandas DataFrame

    Parameters
    ----------
    term_term_matrix : scipy.sparse.spmatrix
        [description]
    id2token : Mapping[int,str]
        [description]
    documents : pd.DataFrame, optional
        [description], by default None
    min_count : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """
    coo_df = (
        pd.DataFrame({'w1_id': term_term_matrix.row, 'w2_id': term_term_matrix.col, 'value': term_term_matrix.data})[
            ['w1_id', 'w2_id', 'value']
        ]
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if min_count > 1:
        coo_df = coo_df[coo_df.value >= min_count]

    if documents is not None:

        coo_df['value_n_d'] = coo_df.value / float(len(documents))

        if 'n_tokens' in documents:
            coo_df['value_n_t'] = coo_df.value / float(sum(documents.n_tokens.values()))

    coo_df['w1'] = coo_df.w1_id.apply(lambda x: id2token[x])
    coo_df['w2'] = coo_df.w2_id.apply(lambda x: id2token[x])

    coo_df = coo_df[['w1', 'w2', 'value', 'value_n_d', 'value_n_t']]

    return coo_df
