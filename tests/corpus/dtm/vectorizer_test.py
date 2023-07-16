import os

import numpy as np
import pandas as pd

import penelope.corpus as pc
from penelope.pipeline.convert import tagged_frame_to_tokens
from tests.fixtures import MockedProcessedCorpus, TranstromerCorpus
from tests.utils import TEST_CORPUS_FILENAME, create_test_corpus_tokens_reader


def mock_corpus() -> MockedProcessedCorpus:
    mock_corpus_data = [
        ('document_2013_1.txt', "a a b c c c c d"),
        ('document_2013_2.txt', "a a b b c c c"),
        ('document_2014_1.txt', "a a b b b c c"),
        ('document_2014_2.txt', "a a b b b b c d"),
        ('document_2014_3.txt', "a a c d"),
    ]
    corpus: MockedProcessedCorpus = MockedProcessedCorpus(mock_corpus_data)
    return corpus


def create_reader():
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_test_corpus_tokens_reader(
        filename_fields=filename_fields, text_transforms="dehyphen,normalize-whitespace"
    )
    return reader


def create_corpus():
    reader = create_reader()
    transform_opts = pc.TokensTransformOpts(
        transforms={'only-any-alphanumeric': True, 'to-lower': True, 'min-chars': 2, 'remove-numerals': True}
    )
    corpus = pc.TokenizedCorpus(reader, transform_opts=transform_opts)
    return corpus


def test_create_text_tokenizer_smoke_test():
    reader = pc.TokenizeTextReader(TEST_CORPUS_FILENAME, reader_opts=pc.TextReaderOpts())
    assert reader is not None
    assert next(reader) is not None


def test_fit_transform_creates_a_vocabulary_with_unique_tokens_with_an_id_sequence():
    corpus = create_corpus()
    vectorizer = pc.CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus, already_tokenized=True)
    assert corpus.token2id == v_corpus.token2id


def test_fit_transform_creates_a_bag_of_word_bag_term_matrix():
    corpus = mock_corpus()
    vectorizer = pc.CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus, already_tokenized=True)
    expected_vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    expected_dtm = [[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]]
    expected_word_counts = {'a': 10, 'b': 10, 'c': 11, 'd': 3}
    assert expected_vocab, v_corpus.token2id
    assert expected_word_counts, v_corpus.term_frequency
    assert (expected_dtm == v_corpus.bag_term_matrix.toarray()).all()


def test_term_frequency_are_absolute_word_of_entire_corpus():
    corpus = create_corpus()
    vectorizer = pc.CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus, already_tokenized=True)
    results = v_corpus.term_frequency
    expected = {
        'tre': 1,
        'svarta': 1,
        'ekar': 1,
        'ur': 2,
        'snön': 1,
        'så': 3,
        'grova': 1,
        'men': 2,
        'fingerfärdiga': 1,
        'deras': 1,
        'väldiga': 2,
        'flaskor': 1,
        'ska': 1,
        'grönskan': 1,
        'skumma': 1,
        'vår': 1,
        'på': 3,
        'väg': 1,
        'det': 3,
        'långa': 1,
        'mörkret': 2,
        'envist': 1,
        'skimrar': 1,
        'mitt': 1,
        'armbandsur': 1,
        'med': 2,
        'tidens': 1,
        'fångna': 1,
        'insekt': 1,
        'nordlig': 1,
        'storm': 1,
        'är': 5,
        'den': 3,
        'tid': 1,
        'när': 1,
        'rönnbärsklasar': 1,
        'mognar': 1,
        'vaken': 1,
        'hör': 1,
        'man': 2,
        'stjärnbilderna': 1,
        'stampa': 1,
        'sina': 1,
        'spiltor': 1,
        'högt': 1,
        'över': 1,
        'trädet': 1,
        'jag': 4,
        'ligger': 1,
        'sängen': 1,
        'armarna': 1,
        'utbredda': 1,
        'ett': 1,
        'ankare': 1,
        'som': 4,
        'grävt': 1,
        'ner': 1,
        'sig': 1,
        'ordentligt': 1,
        'och': 2,
        'håller': 1,
        'kvar': 1,
        'skuggan': 1,
        'flyter': 1,
        'där': 1,
        'ovan': 1,
        'stora': 1,
        'okända': 1,
        'en': 2,
        'del': 1,
        'av': 1,
        'säkert': 1,
        'viktigare': 1,
        'än': 1,
        'har': 2,
        'sett': 1,
        'mycket': 2,
        'verkligheten': 1,
        'tärt': 1,
        'här': 1,
        'sommaren': 1,
        'till': 1,
        'sist': 1,
    }
    assert ([expected[v_corpus.id2token[i]] for i in range(0, len(expected))] == results).all()


def test_fit_transform_when_given_a_vocabulary_returns_same_vocabulary():
    corpus: pc.TokenizedCorpus = pc.TokenizedCorpus(
        reader=create_reader(),
        transform_opts=pc.TokensTransformOpts(transforms={'to-lower': True, 'min-chars': 10}),
    )

    vocabulary = pc.CorpusVectorizer().fit_transform(corpus, already_tokenized=True).token2id

    assert corpus.token2id == vocabulary

    expected_vocabulary_reversed = {k: abs(v - 5) for k, v in corpus.token2id.items()}

    vocabulary = (
        pc.CorpusVectorizer()
        .fit_transform(corpus, already_tokenized=True, vocabulary=expected_vocabulary_reversed)
        .token2id
    )

    assert expected_vocabulary_reversed == vocabulary


def test_dump_of_transtromer_text_corpus():
    folder: str = 'tests/output/tranströmer'
    os.makedirs(folder, exist_ok=True)
    corpus: pc.VectorizedCorpus = pc.CorpusVectorizer().fit_transform(TranstromerCorpus())
    assert corpus is not None
    corpus.dump(tag='tranströmer', folder=folder)
    assert corpus.dump_exists(tag='tranströmer', folder=folder)

def test_dump_of_transtromer_pos_csv_corpus():
    folder: str = 'tests/output/tranströmer'
    os.makedirs(folder, exist_ok=True)
    
    # tagged_frame_to_tokens(  # pylint: disable=too-many-arguments, too-many-statements
    #     doc,
    #     extract_opts=ExtractTaggedTokensOpts(),
    #     token2id=None,
    #     transform_opts = TokensTransformOpts(),
    #     pos_schema = PoS_Tag_Schemes.SUC,
    # )

    corpus: pc.VectorizedCorpus = pc.CorpusVectorizer().fit_transform(TranstromerCorpus())
    assert corpus is not None
    corpus.dump(tag='tranströmer', folder=folder)
    assert corpus.dump_exists(tag='tranströmer', folder=folder)



def test_from_token_ids_stream():
    tokenized_corpus: MockedProcessedCorpus = mock_corpus()
    token2id: dict = tokenized_corpus.token2id
    id2token: dict = {v: k for k, v in tokenized_corpus.token2id.items()}

    """Arrange: simulate tagged ID frame payloads by turning corpus into a stream of document_id ✕ pd.Series"""
    document_index: pd.DataFrame = tokenized_corpus.document_index
    name2id = document_index.set_index('filename')['document_id'].to_dict().get
    tokens2series = lambda tokens: pd.Series([token2id[t] for t in tokens], dtype=np.int64)
    stream = [(name2id(filename), tokens2series(tokens)) for filename, tokens in tokenized_corpus]
    assert [id2token[t] for t in stream[0][1]] == tokenized_corpus.data[0][1]

    """Act: create a vectorized corpus out of stream"""

    vectorized_corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.from_token_id_stream(stream, token2id, document_index)

    assert vectorized_corpus is not None

    """Check results"""
    expected_dtm = np.matrix([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])

    assert (vectorized_corpus.data.todense() == expected_dtm).all()
