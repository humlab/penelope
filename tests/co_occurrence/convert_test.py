import os

import pandas as pd
import penelope.co_occurrence as co_occurrence
import scipy
from penelope.co_occurrence import (
    Bundle,
    co_occurrences_to_co_occurrence_corpus,
    term_term_matrix_to_co_occurrences,
    truncate_by_global_threshold,
)
from penelope.corpus import DocumentIndexHelper, Token2Id, TokenizedCorpus, VectorizedCorpus, dtm
from penelope.type_alias import CoOccurrenceDataFrame, DocumentIndex
from tests.fixtures import (
    SIMPLE_CORPUS_ABCDE_5DOCS,
    very_simple_corpus,
    very_simple_corpus_co_occurrences,
    very_simple_term_term_matrix,
)

jj = os.path.join


def test_to_co_occurrence_matrix():

    text_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)

    term_term_matrix1 = very_simple_term_term_matrix(text_corpus)

    term_term_matrix2 = co_occurrence.to_co_occurrence_matrix(text_corpus)

    assert (term_term_matrix1 != term_term_matrix2).nnz == 0


def test_to_vectorized_corpus():

    """Create an empty Bundle instance to get the filename right"""
    bundle: co_occurrence.Bundle = co_occurrence.Bundle(folder='./tests/test_data/VENUS', tag="VENUS")

    co_occurrences: CoOccurrenceDataFrame = co_occurrence.load_co_occurrences(bundle.co_occurrence_filename)
    document_index: DocumentIndex = DocumentIndexHelper.load(bundle.document_index_filename).document_index
    token2id: Token2Id = Token2Id.load(bundle.dictionary_filename)

    corpus = co_occurrences_to_co_occurrence_corpus(
        co_occurrences=co_occurrences,
        document_index=document_index,
        token2id=token2id,
    )

    assert corpus.data.sum() == co_occurrences.value.sum()
    assert corpus.data.shape[0] == len(document_index)
    assert corpus.data.shape[1] == len(co_occurrences[["w1_id", "w2_id"]].drop_duplicates())


def test_truncate_by_global_threshold2():

    corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        concept={'g'}, ignore_concept=False, context_width=1
    )
    co_occurrences: pd.DataFrame = very_simple_corpus_co_occurrences(corpus, context_opts=context_opts).co_occurrences

    truncated_co_occurrences = truncate_by_global_threshold(co_occurrences=co_occurrences, threshold=1)

    assert truncated_co_occurrences is not None

    # FIXME Add more tests/asserts


def test_term_term_matrix_to_co_occurrences_with_ignore_ids():

    text_corpus = very_simple_corpus(
        data=[
            ('tran_2019_01_test.txt', ['*', 'b', 'c', 'c']),
            ('tran_2019_02_test.txt', ['a', '*', '*', 'd']),
            ('tran_2019_03_test.txt', ['a', 'e', 'e', 'b']),
            ('tran_2020_01_test.txt', ['*', 'c', 'd', 'a']),
            ('tran_2020_02_test.txt', ['a', 'b', '*', '*']),
        ]
    )
    token2id: Token2Id = Token2Id(text_corpus.token2id)

    term_term_matrix = (
        dtm.CorpusVectorizer()
        .fit_transform(text_corpus, already_tokenized=True, vocabulary=text_corpus.token2id)
        .co_occurrence_matrix()
    )

    pad_id = token2id['*']

    co_occurrences = term_term_matrix_to_co_occurrences(
        term_term_matrix=term_term_matrix,
        threshold_count=1,
        ignore_ids=set([pad_id]),
    )

    assert not (co_occurrences.w1_id == pad_id).any()
    assert not (co_occurrences.w2_id == pad_id).any()


def test_term_term_matrix_to_co_occurrences_with_no_ignore_ids():

    text_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    term_term_matrix: scipy.sparse.spmatrix = very_simple_term_term_matrix(text_corpus)

    co_occurrences = term_term_matrix_to_co_occurrences(
        term_term_matrix=term_term_matrix,
        threshold_count=1,
        ignore_ids=None,
    )

    fg = text_corpus.token2id.get
    assert co_occurrences.value.sum() == term_term_matrix.sum()
    assert 4 == int(co_occurrences[((co_occurrences.w1_id == fg('a')) & (co_occurrences.w2_id == fg('c')))].value)
    assert 1 == int(co_occurrences[((co_occurrences.w1_id == fg('b')) & (co_occurrences.w2_id == fg('d')))].value)


def test_co_occurrences_to_co_occurrence_corpus():

    corpus: TokenizedCorpus = very_simple_corpus(SIMPLE_CORPUS_ABCDE_5DOCS)
    context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
        concept={}, ignore_concept=False, context_width=1
    )

    token2id: Token2Id = Token2Id(corpus.token2id)

    value: Bundle = very_simple_corpus_co_occurrences(corpus, context_opts=context_opts)

    corpus = VectorizedCorpus.from_co_occurrences(
        co_occurrences=value.co_occurrences,
        document_index=value.document_index,
        token2id=token2id,
    )
    assert corpus is not None
