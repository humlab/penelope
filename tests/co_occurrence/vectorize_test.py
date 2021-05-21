import collections
from typing import Iterable

import numpy as np
import scipy
from penelope.corpus import CorpusVectorizer, Token2Id, VectorizedCorpus
from penelope.pipeline.co_occurrence.tasks import CoOccurrenceMatrixBundle, TTM_to_coo_DTM
from penelope.type_alias import DocumentIndex
from tests.fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, already_tokenized=True, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id


def test_TTM_to_COO_DTM_using_lil_matrix():

    t_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    t_token2id = Token2Id(t_corpus.token2id)
    t_document_index: DocumentIndex = t_corpus.document_index

    stream: Iterable[CoOccurrenceMatrixBundle] = (
        CoOccurrenceMatrixBundle(
            document_id,
            CorpusVectorizer()
            .fit_transform([doc], already_tokenized=True, vocabulary=t_corpus.token2id)
            .co_occurrence_matrix(),
            collections.Counter(),
        )
        for document_id, doc in enumerate(t_corpus)
    )

    corpus: VectorizedCorpus = TTM_to_coo_DTM(stream, t_token2id, t_document_index)

    assert corpus is not None


def test_development_TTM_to_coo_DTM():

    document_id: int = 0
    # term_term_matrix: scipy.sparse.spmatrix
    token2id: Token2Id = Token2Id()

    t_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    v_corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(
        t_corpus, already_tokenized=True, vocabulary=t_corpus.token2id
    )
    TTM: scipy.sparse.spmatrix = v_corpus.co_occurrence_matrix()

    """Maximum number of tokens is number of elements in upper triangular, diagonal excluded"""
    vocab_size = int(TTM.shape[0] * (TTM.shape[0] - 1) / 2)

    """Ingest token-pairs into new vocabulary"""
    fg = t_corpus.id2token.get
    token2id.ingest(f"{fg(a)}/{fg(b)}" for (a, b) in zip(TTM.row, TTM.col))

    """Translate token-pair ids into id in new vocabulary"""
    token_ids = [token2id[f"{fg(a)}/{fg(b)}"] for (a, b) in zip(TTM.row, TTM.col)]
    document_ids = np.empty(len(token_ids))
    document_ids.fill(document_id)

    shape = (1, vocab_size)

    matrix = scipy.sparse.coo_matrix(
        (
            TTM.data.astype(np.uint16),
            (
                document_ids,  # .astype(np.uint16),
                token_ids,  # .astype(np.uint32),
            ),
        ),
        shape=shape,
    )

    matrix = scipy.sparse.lil_matrix(shape, dtype=int)

    # X = self.term_doc_matrix.get_metadata_doc_mat() if non_text else self.term_doc_matrix.get_term_doc_mat()
    # for i, domain in enumerate(doc_domain_set):
    #     domain_mat[i, :] = X[np.array(doc_domains == domain)].sum(axis=0)
    # return domain_mat.tocsr()

    assert matrix is not None

    # """Convert a sequence of TTM to a CC-Corpus"""
    # token2id.ingest()
