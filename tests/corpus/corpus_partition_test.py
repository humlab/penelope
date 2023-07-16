from penelope.corpus import SparvTokenizedCsvCorpus, TokensTransformOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts, TokenizeTextReader
from tests.pipeline.fixtures import SPARV_TAGGED_COLUMNS

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer/tranströmer_corpus_export.sparv4.csv.zip'


def create_test_corpus() -> SparvTokenizedCsvCorpus:
    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_opts=ExtractTaggedTokensOpts(lemmatize=True, **SPARV_TAGGED_COLUMNS),
        transform_opts=TokensTransformOpts(),
    )

    return corpus


def test_partition_documents():
    expected_groups = {
        2019: ['tran_2019_01_test', 'tran_2019_02_test', 'tran_2019_03_test'],
        2020: ['tran_2020_01_test', 'tran_2020_02_test'],
    }

    groups = create_test_corpus().group_documents_by_key('year')

    assert expected_groups == groups


def test_partition_groups_by_year_contains_year():
    expected_groups = {
        2019: ['tran_2019_01_test', 'tran_2019_02_test', 'tran_2019_03_test'],
    }

    groups = create_test_corpus().group_documents_by_key('year')

    assert expected_groups[2019] == groups[2019]


def test_corpus_apply_when_single_group_partition_filter_then_other_groups_are_filtered_out():
    expected_document_names = ['tran_2019_01_test', 'tran_2019_02_test', 'tran_2019_03_test']

    corpus: SparvTokenizedCsvCorpus = create_test_corpus()

    document_names = corpus.group_documents_by_key('year')[2019]

    assert expected_document_names == document_names
    assert isinstance(corpus.reader, TokenizeTextReader)
    assert hasattr(corpus.reader, 'apply_filter')

    corpus.reader.apply_filter(document_names)

    assert corpus.document_names == document_names

    expected_processed_filenames = [f'{x}.txt' for x in expected_document_names]
    for i, (filename, _) in enumerate(corpus):
        assert expected_processed_filenames[i] == filename


def test_corpus_apply_when_looping_through_partition_groups_filter_outs_other_groups():
    expected_groups = {
        2019: ['tran_2019_01_test', 'tran_2019_02_test', 'tran_2019_03_test'],
        2020: ['tran_2020_01_test', 'tran_2020_02_test'],
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
        reader_opts=TextReaderOpts(
            filename_fields="year:_:1",
        ),
        extract_opts=ExtractTaggedTokensOpts(
            lemmatize=True, pos_includes='|NN|', pos_paddings=None, **SPARV_TAGGED_COLUMNS
        ),
        transform_opts=TokensTransformOpts(transforms={'min-chars': 2, 'to_upper': True}),
    )

    partitions = corpus.group_documents_by_key('year')

    for key in partitions:
        corpus.reader.apply_filter(partitions[key])
        assert expected_groups[key] == corpus.document_names

        tokens = [x for x in corpus.terms]
        assert expected_tokens[key] == tokens
