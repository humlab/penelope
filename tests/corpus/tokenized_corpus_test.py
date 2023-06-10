import penelope.corpus.tokenized_corpus as corpora
from penelope.corpus import TokensTransformOpts
from tests.utils import create_tokens_reader


def create_reader():
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, text_transforms="normalize-whitespace,dehyphen")
    return reader


def test_next_document_when_only_any_alphanumeric_is_false_returns_all_tokens():
    reader = create_reader()
    transform_opts = TokensTransformOpts()
    corpus = corpora.TokenizedCorpus(reader, transform_opts=transform_opts)
    _, tokens = next(corpus)
    expected = (
        "Tre svarta ekar ur snön . Så grova , men fingerfärdiga . Ur deras väldiga flaskor ska grönskan skumma i vår ."
    )
    assert expected.split() == tokens


def test_next_document_when_only_any_alphanumeric_true_skips_deliminators():
    reader = create_reader()
    corpus = corpora.TokenizedCorpus(
        reader,
        transform_opts=TokensTransformOpts(transforms={'only-any-alphanumeric': True}),
    )
    _, tokens = next(corpus)
    expected = "Tre svarta ekar ur snön Så grova men fingerfärdiga Ur deras väldiga flaskor ska grönskan skumma i vår"
    assert expected.split() == tokens


def test_next_document_when_only_any_alphanumeric_true_skips_deliminators_using_defaults():
    reader = create_tokens_reader(filename_fields=None, text_transforms="normalize-whitespace,dehyphen")
    corpus = corpora.TokenizedCorpus(
        reader, transform_opts=TokensTransformOpts(transforms={'only-any-alphanumeric': True})
    )
    _, tokens = next(corpus)
    expected = "Tre svarta ekar ur snön Så grova men fingerfärdiga Ur deras väldiga flaskor ska grönskan skumma i vår"
    assert expected.split() == tokens


def test_next_document_when_to_lower_is_true_returns_all_lowercase():
    reader = create_reader()
    transform_opts = TokensTransformOpts(transforms={'only-any-alphanumeric': True, 'to-lower': True})
    corpus = corpora.TokenizedCorpus(reader, transform_opts=transform_opts)
    _, tokens = next(corpus)
    expected = "tre svarta ekar ur snön så grova men fingerfärdiga ur deras väldiga flaskor ska grönskan skumma i vår"
    assert expected.split() == tokens


def test_next_document_when_min_len_is_two_returns_single_char_words_filtered_out():
    reader = create_reader()
    transform_opts = TokensTransformOpts(transforms={'only-any-alphanumeric': True, 'to-lower': True, 'min-chars': 2})
    corpus = corpora.TokenizedCorpus(reader, transform_opts=transform_opts)
    _, tokens = next(corpus)
    expected = "tre svarta ekar ur snön så grova men fingerfärdiga ur deras väldiga flaskor ska grönskan skumma vår"
    assert expected.split() == tokens


def test_next_document_when_max_len_is_six_returns_filter_out_longer_words():
    reader = create_reader()
    transform_opts = TokensTransformOpts(
        transforms={
            'only-any-alphanumeric': True,
            'to-lower': True,
            'min-chars': 2,
            'max-chars': 6,
        }
    )
    corpus = corpora.TokenizedCorpus(reader, transform_opts=transform_opts)
    _, tokens = next(corpus)
    expected = "tre svarta ekar ur snön så grova men ur deras ska skumma vår"
    assert expected.split() == tokens


def test_n_tokens_when_exhausted_and_only_any_alphanumeric_min_len_two_returns_expected_count():
    reader = create_reader()
    corpus = corpora.TokenizedCorpus(
        reader,
        transform_opts=TokensTransformOpts(transforms={'only-any-alphanumeric': True, 'min-chars': 2}),
    )
    n_expected = [17, 13, 21, 42, 18]
    _ = [x for x in corpus]
    n_tokens = list(corpus.document_index.n_tokens)
    assert n_expected == n_tokens


def test_next_document_when_new_corpus_returns_document():
    reader = create_tokens_reader(text_transforms="normalize-whitespace,dehyphen")
    corpus = corpora.TokenizedCorpus(reader)
    result = next(corpus)
    expected = (
        "Tre svarta ekar ur snön . "
        + "Så grova , men fingerfärdiga . "
        + "Ur deras väldiga flaskor "
        + "ska grönskan skumma i vår ."
    )
    assert expected == ' '.join(result[1])


def test_get_index_when_extract_passed_returns_metadata():
    filename_fields = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = create_tokens_reader(filename_fields=filename_fields, text_transforms="normalize-whitespace,dehyphen")
    corpus = corpora.TokenizedCorpus(reader)
    result = corpus.metadata
    expected = [
        dict(filename='dikt_2019_01_test.txt', serial_no=1, year=2019),
        dict(filename='dikt_2019_02_test.txt', serial_no=2, year=2019),
        dict(filename='dikt_2019_03_test.txt', serial_no=3, year=2019),
        dict(filename='dikt_2020_01_test.txt', serial_no=1, year=2020),
        dict(filename='dikt_2020_02_test.txt', serial_no=2, year=2020),
    ]
    assert len(corpus.document_index) == len(expected)
    for i in range(0, len(expected)):
        assert expected[i] == result[i]


def test_get_index_when_no_extract_passed_returns_not_none():
    reader = create_tokens_reader(filename_fields=None, text_transforms="normalize-whitespace,dehyphen")
    corpus = corpora.TokenizedCorpus(reader)
    result = corpus.metadata
    assert result is not None


def test_next_document_when_token_corpus_returns_tokenized_document():
    reader = create_tokens_reader(filename_fields=None, text_transforms="normalize-whitespace,dehyphen")
    corpus = corpora.TokenizedCorpus(reader, transform_opts=TokensTransformOpts())
    _, tokens = next(corpus)
    expected = (
        "Tre svarta ekar ur snön . Så grova , men fingerfärdiga . Ur deras väldiga flaskor ska grönskan skumma i vår ."
    )
    assert expected.split() == tokens


def test_get_index_when_extract_passed_returns_expected_count():
    reader = create_reader()
    transform_opts = TokensTransformOpts({'min-chars': 2})
    corpus = corpora.TokenizedCorpus(reader, transform_opts=transform_opts)
    result = corpus.metadata
    assert 5 == len(result)


def test_n_tokens_when_exhausted_iterater_returns_expected_count():
    reader = create_reader()
    corpus = corpora.TokenizedCorpus(reader, transform_opts=TokensTransformOpts())
    _ = [x for x in corpus]
    n_tokens = list(corpus.document_index.n_tokens)
    expected = [22, 16, 26, 45, 21]
    assert expected == n_tokens


def test_n_tokens_when_exhausted_and_only_any_alphanumeric_is_true_returns_expected_count():
    reader = create_tokens_reader(filename_fields=None, text_transforms="normalize-whitespace,dehyphen")
    corpus = corpora.TokenizedCorpus(
        reader, transform_opts=TokensTransformOpts(transforms={'only-any-alphanumeric': True})
    )
    _ = [x for x in corpus]
    n_tokens = list(corpus.document_index.n_tokens)
    expected = [18, 14, 24, 42, 18]
    assert expected == n_tokens


def test_corpus_can_be_reiterated():
    reader = create_tokens_reader(filename_fields=None, text_transforms="normalize-whitespace,dehyphen")

    corpus = corpora.TokenizedCorpus(
        reader, transform_opts=TokensTransformOpts(transforms={'only-any-alphanumeric': True})
    )
    for _ in range(0, 4):
        n_tokens = [len(x) for x in corpus.terms]
        expected = [18, 14, 24, 42, 18]
        assert expected == n_tokens  # , f"iteration{i}"
