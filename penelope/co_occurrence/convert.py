from typing import Mapping, Union

import scipy
from penelope.corpus import CorpusVectorizer, ITokenizedCorpus, TokenizedCorpus, VectorizedCorpus
from penelope.corpus.readers import ICorpusReader
from penelope.notebook.word_trends import TrendsData

from .persistence import Bundle


def to_trends_data(bundle: Bundle, n_count=25000):

    trends_data = TrendsData(
        compute_options=bundle.compute_options,
        corpus=bundle.corpus,
        corpus_folder=bundle.corpus_folder,
        corpus_tag=bundle.corpus_tag,
        n_count=n_count,
    ).remember(co_occurrences=bundle.co_occurrences, document_index=bundle.document_index)

    return trends_data


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
    dtm_corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        corpus_or_reader, already_tokenized=True, vocabulary=vocabulary
    )
    term_term_matrix = dtm_corpus.co_occurrence_matrix()
    return term_term_matrix
