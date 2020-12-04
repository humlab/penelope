from typing import Mapping, Union

import pandas as pd
import scipy
from penelope.corpus import CorpusVectorizer, TokenizedCorpus
from penelope.corpus.interfaces import ITokenizedCorpus
from penelope.corpus.readers.interfaces import ICorpusReader
from penelope.utility import getLogger

logger = getLogger()


def to_co_occurrence_matrix(
    corpus_or_reader: Union[ICorpusReader, TokenizedCorpus], vocabulary: Mapping[str, int] = None
) -> scipy.sparse.spmatrix:
    """Computes a term-term co-ocurrence matrix for documents in corpus/reader.

    Parameters
    ----------
    corpus_or_reader : Union[ICorpusReader,TokenizedCorpus]
        Sequence of tokenized documents

    Returns
    -------
    pd.DataFrame
        Upper diagonal of term-term frequency matrix (TTM). Note that diagonal (wi, wi) is not returned
    """

    if not isinstance(corpus_or_reader, ITokenizedCorpus):
        corpus_or_reader = TokenizedCorpus(reader=corpus_or_reader)

    vocabulary = vocabulary or corpus_or_reader.token2id
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus_or_reader, already_tokenized=True, vocabulary=vocabulary)
    term_term_matrix = v_corpus.co_occurrence_matrix()

    return term_term_matrix


def to_dataframe(
    term_term_matrix: scipy.sparse.spmatrix,
    id2token: Mapping[int, str],
    catalogue: pd.DataFrame = None,
    threshold_count: int = 1,
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
    threshold_count : int, optional
        Min count (`value`) to include in result, by default 1

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

    if threshold_count > 0:
        coo_df = coo_df[coo_df.value >= threshold_count]

    if catalogue is not None:

        coo_df['value_n_d'] = coo_df.value / float(len(catalogue))

        for n_token_count, target_field in [('n_tokens', 'value_n_t'), ('n_raw_tokens', 'value_n_r_t')]:

            if n_token_count in catalogue.columns:
                try:
                    coo_df[target_field] = coo_df.value / float(sum(catalogue[n_token_count].values))
                except ZeroDivisionError:
                    coo_df[target_field] = 0.0
            else:
                logger.warning(f"{target_field}: cannot compute since {n_token_count} not in corpus document catalogue")

    coo_df['w1'] = coo_df.w1_id.apply(lambda x: id2token[x])
    coo_df['w2'] = coo_df.w2_id.apply(lambda x: id2token[x])

    coo_df = coo_df[['w1', 'w2', 'value', 'value_n_d', 'value_n_t']]

    return coo_df
