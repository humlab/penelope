import numpy as np
import pandas as pd
import scipy
from penelope.corpus import tokenized_corpus
from penelope.corpus import vectorizer as corpus_vectorizer


def compute_coocurrence_matrix(reader, min_count: int = 1, **kwargs) -> pd.DataFrame:
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

    term_term_matrix = np.dot(v_corpus.bag_term_matrix.T, v_corpus.bag_term_matrix)
    term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)

    id2token = {i: t for t, i in v_corpus.token2id.items()}

    cdf = (
        pd.DataFrame({'w1_id': term_term_matrix.row, 'w2_id': term_term_matrix.col, 'value': term_term_matrix.data})[
            ['w1_id', 'w2_id', 'value']
        ]
        .sort_values(['w1_id', 'w2_id'])
        .reset_index(drop=True)
    )

    if min_count > 1:
        cdf = cdf[cdf.value >= min_count]

    n_documents = len(corpus.metadata)
    n_tokens = sum(corpus.n_raw_tokens.values())

    cdf['value_n_d'] = cdf.value / float(n_documents)
    cdf['value_n_t'] = cdf.value / float(n_tokens)

    cdf['w1'] = cdf.w1_id.apply(lambda x: id2token[x])
    cdf['w2'] = cdf.w2_id.apply(lambda x: id2token[x])

    return cdf[['w1', 'w2', 'value', 'value_n_d', 'value_n_t']]
