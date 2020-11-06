from penelope.corpus.readers.annotation_opts import AnnotationOpts
from penelope.corpus.readers.text_tokenizer import TextTokenizer
from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
from penelope.corpus.tokens_transformer import TokensTransformOpts

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'


def create_test_corpus() -> SparvTokenizedCsvCorpus:

    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        tokenizer_opts=dict(
            filename_fields="year:_:1",
        ),
        annotation_opts=AnnotationOpts(),
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
        2019: [
            [
                'KYRKA',
                'TURIST',
                'HALVMÖRKER',
                'VALV',
                'VALV',
                'ÖVERBLICK',
                'LJUSLÅGA',
                'ÄNGEL',
                'ANSIKTE',
                'KROPP',
                'MÄNNISKA',
                'VALV',
                'VALV',
                'TÅR',
                'PIAZZA',
                'MR',
                'MRS',
                'HERR',
                'SIGNORA',
                'VALV',
                'VALV',
            ],
            [
                'KÖR',
                'NATT',
                'HUS',
                'STRÅLKASTARSKEN',
                'HUS',
                'LADA',
                'FORDON',
                'NU',
                'LIV',
                'MÄNNISKA',
                'DEL',
                'ANLETSDRAG',
                'TRÄNING',
                'EVIGHET',
                'ALLT',
                'SÖMN',
                'BOM',
                'MYSTERIUM',
            ],
            [
                'SKOG',
                'GLÄNTA',
                'GLÄNTA',
                'OMSLUT',
                'SKOG',
                'SJÄLV',
                'STAM',
                'LAV',
                'SKÄGGSTUBB',
                'TRÄD',
                'TOPP',
                'KVIST',
                'LJUS',
                'SKUGGA',
                'SKUGGA',
                'KÄRR',
                'PLATS',
                'GRÄS',
                'STEN',
                'VARA',
                'GRUNDSTEN',
                'HUS',
                'HÄR',
                'UPPLYSNING',
                'NAMN',
                'ARKIV',
                'ARKIV',
                'TRADITION',
                'DÖD',
                'MINNE',
                'ZIGENARSTAMMEN',
                'MEN',
                'TORP',
                'RÖST',
                'VÄRLD',
                'CENTRUM',
                'INVÅNARE',
                'KRÖNIKA',
                'ÖDE',
                'ÅR',
                'TORP',
                'SFINX',
                'GRUNDSTEN',
                'SÄTT',
                'MÅSTE',
                'NU',
                'SNÅR',
                'SIDA',
                'STEG',
                'GÅNGSTIG',
                'KOMMUNIKATIONSNÄT',
                'KRAFTLEDNINGSSTOLPEN',
                'SKALBAGGE',
                'SOL',
                'SKÖLD',
                'FLYGVINGARNA',
                'FALLSKÄRM',
                'EXPERT',
            ],
        ],
        2020: [
            [
                'VRAK',
                'KRETSANDE',
                'PUNKT',
                'STILLHET',
                'HAV',
                'LJUS',
                'BETSEL',
                'TÅNG',
                'STRAND',
                'JORD',
                'MÖRKER',
                'FLADDERMUS',
                'VRAK',
                'STJÄRNA',
            ],
            [
                'ÅR',
                'STÖVEL',
                'SOL',
                'TRÄD',
                'VIND',
                'FRIHET',
                'BERG',
                'FOT',
                'BARRSKOGSBRÄNNINGEN',
                'MEN',
                'SOMMAR',
                'DYNING',
                'TRÄD',
                'TOPP',
                'ÖGONBLICK',
                'KUST',
            ],
        ],
    }

    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        tokenizer_opts=dict(
            filename_fields="year:_:1",
        ),
        annotation_opts=AnnotationOpts(pos_includes='|NN|'),
        tokens_transform_opts=TokensTransformOpts(
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
