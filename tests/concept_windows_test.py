import json

from penelope.co_occurrence import ContextOpts, WindowsCorpus, corpus_to_windows
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus

from .test_data.windows_test_data import TRANSTRÖMMER_CORPUS_NNVB_LEMMA, TRANSTRÖMMER_NNVB_LEMMA_WINDOWS

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'


def log_json(filename, d):
    with open(filename, "w", encoding="utf8") as f:
        json.dump(d, f, ensure_ascii=False, indent=True)


def log_corpus_windows_fixture(filename, corpus, windows):
    log_json(filename, {"corpus": [x for x in corpus], "windows": windows})


def load_corpus_windows_fixture(filename: str):
    with open(filename, "r", encoding="utf8") as f:
        return json.load(f)


def test_windowed_when_nn_vb_lemma_2_tokens():

    expected_windows = [
        ["tran_2019_01_test.txt", 0, ["*", "*", "kyrka", "tränga", "turist"]],
        ["tran_2019_01_test.txt", 1, ["turist", "halvmörker", "valv", "gapa", "valv"]],
        ["tran_2019_01_test.txt", 2, ["valv", "gapa", "valv", "överblick", "ljuslåga"]],
        ["tran_2019_01_test.txt", 3, ["människa", "öppna", "valv", "valv", "bli"]],
        ["tran_2019_01_test.txt", 4, ["öppna", "valv", "valv", "bli", "vara"]],
        ["tran_2019_01_test.txt", 5, ["skola", "tår", "piazza", "Mr", "Mrs"]],
        ["tran_2019_01_test.txt", 6, ["signora", "öppna", "valv", "valv", "*"]],
        ["tran_2019_01_test.txt", 7, ["öppna", "valv", "valv", "*", "*"]],
    ]

    corpus = TRANSTRÖMMER_CORPUS_NNVB_LEMMA
    concept = {'piazza', 'kyrka', 'valv'}

    windows = [
        w
        for w in corpus_to_windows(
            stream=corpus, context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=2), pad='*'
        )
    ]

    assert expected_windows == windows


def test_windowed_when_nn_vb_lemma_5_tokens():

    expected_windows = TRANSTRÖMMER_NNVB_LEMMA_WINDOWS
    corpus = TRANSTRÖMMER_CORPUS_NNVB_LEMMA
    concept = {'piazza', 'kyrka', 'valv'}

    windows = [
        w
        for w in corpus_to_windows(
            stream=corpus, context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=5), pad='*'
        )
    ]

    assert expected_windows == windows


def test_windowed_corpus_when_nn_vb_lemma_x_tokens():

    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        reader_opts=TextReaderOpts(),
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|VB|', lemmatize=True),
    )
    expected_windows = TRANSTRÖMMER_NNVB_LEMMA_WINDOWS

    concept = {'piazza', 'kyrka', 'valv'}
    windows = [
        w
        for w in corpus_to_windows(
            stream=corpus, context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=5), pad='*'
        )
    ]

    assert expected_windows == windows


def test_windowed_corpus_when_nn_vb_not_lemma_2_tokens():

    extract_tokens_opts = ExtractTaggedTokensOpts(pos_includes='|NN|VB|', lemmatize=False)
    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME, reader_opts=TextReaderOpts(), extract_tokens_opts=extract_tokens_opts
    )
    expected_windows = [
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

    concept = {'är'}
    windows = [
        w
        for w in corpus_to_windows(
            stream=corpus, context_opts=ContextOpts(concept=concept, ignore_concept=False, context_width=2), pad='*'
        )
    ]

    assert expected_windows == windows
    assert corpus.document_index is not None


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
