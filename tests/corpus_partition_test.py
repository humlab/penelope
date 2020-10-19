from penelope.corpus.readers.text_tokenizer import TextTokenizer
import pytest  # pylint: disable=unused-import

from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'


def create_test_corpus() -> SparvTokenizedCsvCorpus:

    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        tokenizer_opts=dict(filename_fields="year:_:1", ),
    )

    return corpus


def test_partition_documents():

    expected_groups = {
        2019: ['tran_2019_01_test.csv', 'tran_2019_02_test.csv', 'tran_2019_03_test.csv'],
        2020: ['tran_2020_01_test.csv', 'tran_2020_02_test.csv'],
    }

    groups = create_test_corpus().partition_documents('year')

    assert expected_groups == groups


def test_partition_groups_by_year_contains_year():

    expected_groups = {
        2019: ['tran_2019_01_test.csv', 'tran_2019_02_test.csv', 'tran_2019_03_test.csv'],
    }

    groups = create_test_corpus().partition_documents('year')

    assert expected_groups[2019] == groups[2019]


def test_corpus_apply_when_single_group_partition_filter_then_other_groups_are_filtered_out():

    expected_filenames = ['tran_2019_01_test.csv', 'tran_2019_02_test.csv', 'tran_2019_03_test.csv']

    corpus = create_test_corpus()

    filenames = corpus.partition_documents('year')[2019]

    assert expected_filenames == filenames
    assert isinstance(corpus.reader, TextTokenizer)
    assert hasattr(corpus.reader, 'apply_filter')

    corpus.reader.apply_filter(filenames)

    assert corpus.filenames == filenames

    expected_processed_filenames = [x.replace('.csv', '.txt') for x in expected_filenames]
    for i, (filename, _) in enumerate(corpus):
        assert expected_processed_filenames[i] == filename


def test_corpus_apply_when_looping_through_partition_groups_filter_outs_other_groups():

    expected_groups = {
        2019: ['tran_2019_01_test.csv', 'tran_2019_02_test.csv', 'tran_2019_03_test.csv'],
        2020: ['tran_2020_01_test.csv', 'tran_2020_02_test.csv'],
    }

    expected_tokens = {
        2019: [[
            'kyrka', 'turist', 'halvmörker', 'valv', 'valv', 'överblick', 'ljuslåga', 'ängel', 'ansikte', 'kropp',
            'människa', 'valv', 'valv', 'tår', 'piazza', 'mr', 'mrs', 'herr', 'signora', 'valv', 'valv'
        ],
               [
                   'kör', 'natt', 'hus', 'strålkastarsken', 'hus', 'lada', 'fordon', 'nu', 'liv', 'människa', 'del',
                   'anletsdrag', 'träning', 'evighet', 'allt', 'sömn', 'bom', 'mysterium'
               ],
               [
                   'skog', 'glänta', 'glänta', 'omslut', 'skog', 'själv', 'stam', 'lav', 'skäggstubb', 'träd', 'topp',
                   'kvist', 'ljus', 'skugga', 'skugga', 'kärr', 'plats', 'gräs', 'sten', 'vara', 'grundsten', 'hus',
                   'här', 'upplysning', 'namn', 'arkiv', 'arkiv', 'tradition', 'död', 'minne', 'zigenarstammen', 'men',
                   'torp', 'röst', 'värld', 'centrum', 'invånare', 'krönika', 'öde', 'år', 'torp', 'sfinx', 'grundsten',
                   'sätt', 'måste', 'nu', 'snår', 'sida', 'steg', 'gångstig', 'kommunikationsnät',
                   'kraftledningsstolpen', 'skalbagge', 'sol', 'sköld', 'flygvingarna', 'fallskärm', 'expert'
               ]],
        2020: [[
            'vrak', 'kretsande', 'punkt', 'stillhet', 'hav', 'ljus', 'betsel', 'tång', 'strand', 'jord', 'mörker',
            'fladdermus', 'vrak', 'stjärna'
        ],
               [
                   'år', 'stövel', 'sol', 'träd', 'vind', 'frihet', 'berg', 'fot', 'barrskogsbränningen', 'men',
                   'sommar', 'dyning', 'träd', 'topp', 'ögonblick', 'kust'
               ]],
    }

    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        pos_includes='|NN|',
        tokenizer_opts=dict(filename_fields="year:_:1", ),
        tokens_transform_opts=dict(
            min_len=2,
            to_upper=True,
        ),
    )

    partitions = corpus.partition_documents('year')

    for key in partitions:

        corpus.reader.apply_filter(partitions[key])
        assert expected_groups[key] == corpus.filenames

        tokens = [x for x in corpus.terms]
        assert expected_tokens[key] == tokens
