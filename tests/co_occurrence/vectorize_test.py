import numpy as np
import scipy
from penelope.corpus import Token2Id, VectorizedCorpus, CorpusVectorizer
from tests.fixtures import SIMPLE_CORPUS_ABCDE_5DOCS, very_simple_corpus


def test_co_occurrence_matrix_of_corpus_returns_correct_result():

    expected_token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
    expected_matrix = np.matrix([[0, 6, 4, 3, 3], [0, 0, 2, 1, 4], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])

    corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    v_corpus = CorpusVectorizer().fit_transform(corpus, already_tokenized=True, vocabulary=corpus.token2id)

    term_term_matrix = v_corpus.co_occurrence_matrix()

    assert (term_term_matrix.todense() == expected_matrix).all()
    assert expected_token2id == v_corpus.token2id



def test_TTM_to_COODTM():

    # document_id: int
    # term_term_matrix: scipy.sparse.spmatrix
    # token2id: Token2Id = Token2Id()

    t_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    v_corpus: VectorizedCorpus = CorpusVectorizer().fit_transform(t_corpus, already_tokenized=True, vocabulary=corpus.token2id)
    TTM: scipy.sparse.spmatrix = v_corpus.co_occurrence_matrix()

    assert TTM is not None

    # """Convert a sequence of TTM to a CC-Corpus"""
    # token2id.ingest()
    # w1_id = term_term_matrix.row
    # w2_id = term_term_matrix.col
    # values = term_term_matrix.data

    # shape = (len(document_index), len(vocabulary))
    # matrix = scipy.sparse.coo_matrix(
    #     (
    #         co_occurrences.value.astype(np.uint16),
    #         (
    #             co_occurrences.document_id.astype(np.uint32),
    #             token_ids.astype(np.uint32),
    #         ),
    #     ),
    #     shape=shape,
    # )
