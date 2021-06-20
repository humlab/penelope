import json
from typing import Iterable

from penelope.co_occurrence import ContextOpts, WindowsCorpus  # , corpus_to_concept_windows, tokens_to_concept_windows
from penelope.co_occurrence.windows import generate_windows
from penelope.corpus import CorpusVectorizer, generate_token2id
from penelope.type_alias import Token
from tests.fixtures import SAMPLE_WINDOW_STREAM

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.sparv4.csv.zip'


def log_json(filename, d):
    with open(filename, "w", encoding="utf8") as f:
        json.dump(d, f, ensure_ascii=False, indent=True)


def log_corpus_windows_fixture(filename, corpus, windows):
    log_json(filename, {"corpus": [x for x in corpus], "windows": windows})


def load_corpus_windows_fixture(filename: str):
    with open(filename, "r", encoding="utf8") as f:
        return json.load(f)


def test_tokens_to_windows():

    tokens: Iterable[Token] = ["a", "*", "c", "a", "e", "*", "*", "h"]

    context_opts: ContextOpts = ContextOpts(
        concept=set(), context_width=1, ignore_padding=False, pad="*", min_window_size=0
    )
    windows: Iterable[Iterable[str]] = generate_windows(tokens=tokens, context_opts=context_opts)
    expected_windows = [
        ['*', 'a', '*'],
        ['a', '*', 'c'],
        ['*', 'c', 'a'],
        ['c', 'a', 'e'],
        ['a', 'e', '*'],
        ['e', '*', '*'],
        ['*', '*', 'h'],
        ['*', 'h', '*'],
    ]
    assert list(windows) == expected_windows


def test_windows_iterator():

    windows = [
        ['tran_2019_01_test.txt', 0, ['kroppen', 'Skäms', 'är', 'människa', 'öppnar']],
        ['tran_2019_01_test.txt', 1, ['valv', 'blir', 'är', 'skall', 'tårar']],
        ['tran_2019_02_test.txt', 0, ['stiger', 'strålkastarskenet', 'är', 'vill', 'dricka']],
        ['tran_2019_02_test.txt', 1, ['skyltar', 'fordon', 'är', 'nu', 'ikläder']],
        ['tran_2019_02_test.txt', 2, ['allt', 'sömn', 'är', 'vilar', 'bommar']],
        ['tran_2019_03_test.txt', 0, ['gått', 'Gläntan', 'är', 'omsluten', 'skog']],
        ['tran_2019_03_test.txt', 1, ['sammanskruvade', 'träden', 'är', 'ända', 'topparna']],
        ['tran_2019_03_test.txt', 2, ['öppna', 'platsen', 'är', 'gräset', 'ligger']],
        ['tran_2019_03_test.txt', 3, ['arkiv', 'öppnar', 'är', 'arkiven', 'håller']],
        ['tran_2019_03_test.txt', 4, ['håller', 'traditionen', 'är', 'död', 'minnena']],
        ['tran_2019_03_test.txt', 5, ['sorlar', 'röster', 'är', 'världens', 'centrum']],
        ['tran_2019_03_test.txt', 6, ['blir', 'sfinx', 'är', 'grundstenarna', 'sätt']],
        ['tran_2019_03_test.txt', 7, ['gångstig', 'smyger', 'är', 'kommunikationsnätet', 'kraftledningsstolpen']],
    ]

    windows_iter = WindowsCorpus(windows)
    _ = [x for x in windows_iter]
    document_index = windows_iter.document_index

    assert int(document_index[document_index.filename == 'tran_2019_01_test.txt']['n_windows']) == 2
    assert int(document_index[document_index.filename == 'tran_2019_03_test.txt']['n_windows']) == 8
    assert int(document_index[document_index.filename == 'tran_2019_01_test.txt']['n_tokens']) == 10


def test_co_occurrence_given_windows_and_vocabulary_succeeds():

    vocabulary = generate_token2id([x[2] for x in SAMPLE_WINDOW_STREAM])

    windows_corpus = WindowsCorpus(SAMPLE_WINDOW_STREAM, vocabulary=vocabulary)

    v_corpus = CorpusVectorizer().fit_transform(windows_corpus, already_tokenized=True, vocabulary=vocabulary)

    coo_matrix = v_corpus.co_occurrence_matrix()

    assert 10 == coo_matrix.todense()[vocabulary['b'], vocabulary['a']]
    assert 1 == coo_matrix.todense()[vocabulary['d'], vocabulary['c']]
