from penelope.corpus import CorpusVectorizer, TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import TextReaderOpts, TextTokenizer

from tests import utils as test_utils


def mock_corpus():
    mock_corpus_data = [
        ('document_2013_1.txt', "a a b c c c c d"),
        ('document_2013_2.txt', "a a b b c c c"),
        ('document_2014_1.txt', "a a b b b c c"),
        ('document_2014_2.txt', "a a b b b b c d"),
        ('document_2014_2.txt', "a a c d"),
    ]
    corpus = test_utils.MockedProcessedCorpus(mock_corpus_data)
    return corpus


def create_reader():
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = test_utils.create_tokens_reader(
        filename_fields=filename_fields, fix_whitespaces=True, fix_hyphenation=True
    )
    return reader


def create_corpus():
    reader = create_reader()
    tokens_transform_opts = TokensTransformOpts(
        only_any_alphanumeric=True,
        to_lower=True,
        remove_accents=False,
        min_len=2,
        max_len=None,
        keep_numerals=False,
    )
    corpus = TokenizedCorpus(reader, tokens_transform_opts=tokens_transform_opts)
    return corpus


def test_create_text_tokenizer_smoke_test():
    reader = TextTokenizer(test_utils.TEST_CORPUS_FILENAME, reader_opts=TextReaderOpts())
    assert reader is not None
    assert next(reader) is not None


def test_fit_transform_creates_a_vocabulary_with_unique_tokens_with_an_id_sequence():
    corpus = create_corpus()
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus, already_tokenized=True)
    assert corpus.token2id == v_corpus.token2id


def test_fit_transform_creates_a_bag_of_word_bag_term_matrix():
    corpus = mock_corpus()
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus, already_tokenized=True)
    expected_vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    expected_dtm = [[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]]
    expected_word_counts = {'a': 10, 'b': 10, 'c': 11, 'd': 3}
    assert expected_vocab, v_corpus.token2id
    assert expected_word_counts, v_corpus.word_counts
    assert (expected_dtm == v_corpus.bag_term_matrix.toarray()).all()


def test_word_counts_of_vectorized_corpus_are_absolute_word_of_entire_corpus():

    corpus = create_corpus()
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus, already_tokenized=True)
    results = v_corpus.word_counts
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
    assert expected == results


def test_fit_transform_when_given_a_vocabulary_returns_same_vocabulary():

    corpus = TokenizedCorpus(
        reader=create_reader(),
        tokens_transform_opts=TokensTransformOpts(to_lower=True, min_len=10),
    )

    vocabulary = CorpusVectorizer().fit_transform(corpus, already_tokenized=True).token2id

    assert corpus.token2id == vocabulary

    expected_vocabulary_reversed = {k: abs(v - 5) for k, v in corpus.token2id.items()}

    vocabulary = (
        CorpusVectorizer()
        .fit_transform(corpus, already_tokenized=True, vocabulary=expected_vocabulary_reversed)
        .token2id
    )

    assert expected_vocabulary_reversed == vocabulary
