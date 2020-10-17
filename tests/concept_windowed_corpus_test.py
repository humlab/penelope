import json
from penelope.corpus import readers

import pytest  # pylint: disable=unused-import

from penelope.corpus.sparv_corpus import SparvTokenizedCsvCorpus
from penelope.corpus.windowed_corpus import corpus_concept_windows

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'


def log_json(filename, d):
    with open(filename, "w", encoding="utf8") as f:
        json.dump(d, f, ensure_ascii=False, indent=True)


def log_corpus_windows_fixture(filename, corpus, windows):
    log_json(filename, {"corpus": [x for x in corpus], "windows": windows})


def load_corpus_windows_fixture(filename: str):
    with open(filename, "r", encoding="utf8") as f:
        return json.load(f)


FIXED_CORPUS_NNVB_LEMMA = [
    [
        "tran_2019_01_test.txt",
        [
            "kyrka",
            "tränga",
            "turist",
            "halvmörker",
            "valv",
            "gapa",
            "valv",
            "överblick",
            "ljuslåga",
            "fladdra",
            "ängel",
            "ansikte",
            "omfamna",
            "viska",
            "kropp",
            "skämmas",
            "vara",
            "människa",
            "öppna",
            "valv",
            "valv",
            "bli",
            "vara",
            "skola",
            "tår",
            "piazza",
            "Mr",
            "Mrs",
            "herr",
            "signora",
            "öppna",
            "valv",
            "valv",
        ],
    ],
    [
        "tran_2019_02_test.txt",
        [
            "kör",
            "natt",
            "hus",
            "stiga",
            "strålkastarsken",
            "vara",
            "vilja",
            "dricka",
            "hus",
            "lada",
            "skylta",
            "fordon",
            "vara",
            "nu",
            "iklä",
            "liv",
            "människa",
            "sova",
            "del",
            "kunna",
            "sova",
            "ha",
            "anletsdrag",
            "träning",
            "föra",
            "evighet",
            "våga",
            "släppa",
            "allt",
            "sömn",
            "vara",
            "vila",
            "bom",
            "mysterium",
            "dra",
        ],
    ],
    [
        "tran_2019_03_test.txt",
        [
            "finna",
            "skog",
            "glänta",
            "kunna",
            "hitta",
            "gå",
            "glänta",
            "vara",
            "omslut",
            "skog",
            "kväva",
            "själv",
            "stam",
            "lav",
            "skäggstubb",
            "sammanskruvade",
            "träd",
            "vara",
            "ända",
            "topp",
            "kvist",
            "vidröra",
            "ljus",
            "skugga",
            "ruva",
            "skugga",
            "kärr",
            "växa",
            "öppna",
            "plats",
            "vara",
            "gräs",
            "ligga",
            "sten",
            "måste",
            "vara",
            "grundsten",
            "hus",
            "leva",
            "här",
            "kunna",
            "ge",
            "upplysning",
            "namn",
            "finna",
            "arkiv",
            "öppna",
            "vara",
            "arkiv",
            "hålla",
            "tradition",
            "vara",
            "död",
            "minne",
            "Zigenarstammen",
            "minna",
            "men",
            "glömma",
            "anteckna",
            "glömma",
            "torp",
            "sorla",
            "röst",
            "vara",
            "värld",
            "centrum",
            "invånare",
            "dö",
            "flytta",
            "krönika",
            "upphöra",
            "stå",
            "öde",
            "år",
            "torp",
            "bli",
            "sfinx",
            "vara",
            "grundsten",
            "sätt",
            "ha",
            "vara",
            "måste",
            "gå",
            "nu",
            "dyka",
            "snår",
            "gå",
            "tränga",
            "stiga",
            "sida",
            "glesna",
            "ljusna",
            "steg",
            "bli",
            "gångstig",
            "smyga",
            "vara",
            "kommunikationsnät",
            "kraftledningsstolpen",
            "sitta",
            "skalbagge",
            "sol",
            "sköld",
            "ligga",
            "flygvingarna",
            "hopvecklade",
            "fallskärm",
            "expert",
        ],
    ],
    [
        "tran_2020_01_test.txt",
        [
            "vrak",
            "kretsande",
            "punkt",
            "stillhet",
            "rulla",
            "hav",
            "ljus",
            "tugga",
            "betsel",
            "tång",
            "frusta",
            "strand",
            "jord",
            "hölja",
            "mörker",
            "fladdermus",
            "pejla",
            "vrak",
            "stanna",
            "bli",
            "stjärna",
        ],
    ],
    [
        "tran_2020_02_test.txt",
        [
            "år",
            "sparka",
            "stövel",
            "sol",
            "klänga",
            "lövas",
            "träd",
            "fylla",
            "vind",
            "segla",
            "frihet",
            "berg",
            "fot",
            "stå",
            "barrskogsbränningen",
            "men",
            "sommar",
            "dyning",
            "komma",
            "dra",
            "träd",
            "topp",
            "vila",
            "ögonblick",
            "sjunka",
            "kust",
            "stå",
        ],
    ],
]


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

    corpus = FIXED_CORPUS_NNVB_LEMMA
    concept = {'piazza', 'kyrka', 'valv'}

    windows = [w for w in corpus_concept_windows(corpus, concept, 2, pad='*')]

    assert expected_windows == windows
    # # log_corpus_windows_fixture('./tests/test_data/corpus_windows_2_nnvb_lemma.json', corpus, windows)
    # corpus = SparvTokenizedCsvCorpus(SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=True)


def test_windowed_when_nn_vb_lemma_5_tokens():

    expected_windows = [
        [
            "tran_2019_01_test.txt",
            0,
            ["*", "*", "*", "*", "*", "kyrka", "tränga", "turist", "halvmörker", "valv", "gapa"],
        ],
        [
            "tran_2019_01_test.txt",
            1,
            [
                "*",
                "kyrka",
                "tränga",
                "turist",
                "halvmörker",
                "valv",
                "gapa",
                "valv",
                "överblick",
                "ljuslåga",
                "fladdra",
            ],
        ],
        [
            "tran_2019_01_test.txt",
            2,
            [
                "tränga",
                "turist",
                "halvmörker",
                "valv",
                "gapa",
                "valv",
                "överblick",
                "ljuslåga",
                "fladdra",
                "ängel",
                "ansikte",
            ],
        ],
        [
            "tran_2019_01_test.txt",
            3,
            ["kropp", "skämmas", "vara", "människa", "öppna", "valv", "valv", "bli", "vara", "skola", "tår"],
        ],
        [
            "tran_2019_01_test.txt",
            4,
            ["skämmas", "vara", "människa", "öppna", "valv", "valv", "bli", "vara", "skola", "tår", "piazza"],
        ],
        [
            "tran_2019_01_test.txt",
            5,
            ["valv", "bli", "vara", "skola", "tår", "piazza", "Mr", "Mrs", "herr", "signora", "öppna"],
        ],
        ["tran_2019_01_test.txt", 6, ["Mr", "Mrs", "herr", "signora", "öppna", "valv", "valv", "*", "*", "*", "*"]],
        ["tran_2019_01_test.txt", 7, ["Mrs", "herr", "signora", "öppna", "valv", "valv", "*", "*", "*", "*", "*"]],
    ]

    corpus = FIXED_CORPUS_NNVB_LEMMA
    concept = {'piazza', 'kyrka', 'valv'}

    windows = [w for w in corpus_concept_windows(corpus, concept, 5, pad='*')]

    assert expected_windows == windows
    # # log_corpus_windows_fixture('./tests/test_data/corpus_windows_2_nnvb_lemma.json', corpus, windows)
    # corpus = SparvTokenizedCsvCorpus(SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=True)


def test_windowed_corpus_when_nn_vb_lemma_x_tokens():

    corpus = SparvTokenizedCsvCorpus(SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=True)
    expected_windows = [
        [
            'tran_2019_01_test.txt', 0,
            ['*', '*', '*', '*', '*', 'kyrka', 'tränga', 'turist', 'halvmörker', 'valv', 'gapa']
        ],
        [
            'tran_2019_01_test.txt', 1,
            [
                '*', 'kyrka', 'tränga', 'turist', 'halvmörker', 'valv', 'gapa', 'valv', 'överblick', 'ljuslåga',
                'fladdra'
            ]
        ],
        [
            'tran_2019_01_test.txt', 2,
            [
                'tränga', 'turist', 'halvmörker', 'valv', 'gapa', 'valv', 'överblick', 'ljuslåga', 'fladdra', 'ängel',
                'ansikte'
            ]
        ],
        [
            'tran_2019_01_test.txt', 3,
            ['kropp', 'skämmas', 'vara', 'människa', 'öppna', 'valv', 'valv', 'bli', 'vara', 'skola', 'tår']
        ],
        [
            'tran_2019_01_test.txt', 4,
            ['skämmas', 'vara', 'människa', 'öppna', 'valv', 'valv', 'bli', 'vara', 'skola', 'tår', 'piazza']
        ],
        [
            'tran_2019_01_test.txt', 5,
            ['valv', 'bli', 'vara', 'skola', 'tår', 'piazza', 'Mr', 'Mrs', 'herr', 'signora', 'öppna']
        ], ['tran_2019_01_test.txt', 6, ['Mr', 'Mrs', 'herr', 'signora', 'öppna', 'valv', 'valv', '*', '*', '*', '*']],
        ['tran_2019_01_test.txt', 7, ['Mrs', 'herr', 'signora', 'öppna', 'valv', 'valv', '*', '*', '*', '*', '*']]
    ]

    concept = {'piazza', 'kyrka', 'valv'}
    windows = [w for w in corpus_concept_windows(corpus, concept, 5, pad='*')]

    assert expected_windows == windows


def test_windowed_corpus_when_nn_vb_not_lemma_2_tokens():

    corpus = SparvTokenizedCsvCorpus(SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=False)
    expected_windows = [['tran_2019_01_test.txt', 0, ['kroppen', 'Skäms', 'är', 'människa', 'öppnar']],
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
                        [
                            'tran_2019_03_test.txt', 7,
                            ['gångstig', 'smyger', 'är', 'kommunikationsnätet', 'kraftledningsstolpen']
                        ]]

    concept = {'är'}
    windows = [w for w in corpus_concept_windows(corpus, concept, 2, pad='*')]

    assert expected_windows == windows

    # def test_partioned_corpus():

    #     corpus = SparvTokenizedCsvCorpus(SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=False)

    #     partitions = partioned_corpus(corpus.documents, partion_column='year')
    assert corpus.documents is not None


def test_partition_documents():
    # from typing import Callable, Union
    # import pandas as pd

    #tokenizer = readers.SparvCsvTokenizer( SPARV_ZIPPED_CSV_EXPORT_FILENAME, pos_includes='|NN|VB|', lemmatize=True )

    documents = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        tokenizer_opts=dict(filename_fields={'year': r"tran\_(\d{4}).*"}),
    ).documents

    groups = documents.groupby('year')['filename'].aggregate(list).to_dict()

    assert groups == []
    # def partion_document(documents: pd.DataFrame, by: Union[str,Callable]):
    #     raise NotImplementedError()
    #     data = documents.groupby(by=by)