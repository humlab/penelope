import pandas as pd
import tqdm
from penelope.corpus import SparvTokenizedCsvCorpus, TokensTransformOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts, TextReaderOpts

SPARV_ZIPPED_CSV_EXPORT_FILENAME = './tests/test_data/tranströmer_corpus_export.csv.zip'


def test_tokenize_when_nn_lemmatized_lower_returns_correct_tokens():

    expected = [
        (
            'tran_2019_01_test.txt',
            [
                'kyrka',
                'turist',
                'halvmörker',
                'valv',
                'valv',
                'överblick',
                'ljuslåga',
                'ängel',
                'ansikte',
                'kropp',
                'människa',
                'valv',
                'valv',
                'tår',
                'piazza',
                'mr',
                'mrs',
                'herr',
                'signora',
                'valv',
                'valv',
            ],
        ),
        (
            'tran_2019_02_test.txt',
            [
                'kör',
                'natt',
                'hus',
                'strålkastarsken',
                'hus',
                'lada',
                'fordon',
                'nu',
                'liv',
                'människa',
                'del',
                'anletsdrag',
                'träning',
                'evighet',
                'allt',
                'sömn',
                'bom',
                'mysterium',
            ],
        ),
        (
            'tran_2019_03_test.txt',
            [
                'skog',
                'glänta',
                'glänta',
                'omslut',
                'skog',
                'själv',
                'stam',
                'lav',
                'skäggstubb',
                'träd',
                'topp',
                'kvist',
                'ljus',
                'skugga',
                'skugga',
                'kärr',
                'plats',
                'gräs',
                'sten',
                'vara',
                'grundsten',
                'hus',
                'här',
                'upplysning',
                'namn',
                'arkiv',
                'arkiv',
                'tradition',
                'död',
                'minne',
                'zigenarstammen',
                'men',
                'torp',
                'röst',
                'värld',
                'centrum',
                'invånare',
                'krönika',
                'öde',
                'år',
                'torp',
                'sfinx',
                'grundsten',
                'sätt',
                'måste',
                'nu',
                'snår',
                'sida',
                'steg',
                'gångstig',
                'kommunikationsnät',
                'kraftledningsstolpen',
                'skalbagge',
                'sol',
                'sköld',
                'flygvingarna',
                'fallskärm',
                'expert',
            ],
        ),
        (
            'tran_2020_01_test.txt',
            [
                'vrak',
                'kretsande',
                'punkt',
                'stillhet',
                'hav',
                'ljus',
                'betsel',
                'tång',
                'strand',
                'jord',
                'mörker',
                'fladdermus',
                'vrak',
                'stjärna',
            ],
        ),
        (
            'tran_2020_02_test.txt',
            [
                'år',
                'stövel',
                'sol',
                'träd',
                'vind',
                'frihet',
                'berg',
                'fot',
                'barrskogsbränningen',
                'men',
                'sommar',
                'dyning',
                'träd',
                'topp',
                'ögonblick',
                'kust',
            ],
        ),
    ]

    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|NN|', lemmatize=True),
        tokens_transform_opts=TokensTransformOpts(to_lower=True),
    )

    for i, (filename, tokens) in enumerate(corpus):

        assert filename == expected[i][0]
        assert tokens == expected[i][1]


def test_tokenize_when_vb_lemmatized_upper_returns_correct_tokens():

    expected = [
        (
            'tran_2019_01_test.txt',
            [
                'tränga',
                'gapa',
                'fladdra',
                'omfamna',
                'viska',
                'skämmas',
                'vara',
                'öppna',
                'bli',
                'vara',
                'skola',
                'öppna',
            ],
        ),
        (
            'tran_2019_02_test.txt',
            [
                'stiga',
                'vara',
                'vilja',
                'dricka',
                'skylta',
                'vara',
                'iklä',
                'sova',
                'kunna',
                'sova',
                'ha',
                'föra',
                'våga',
                'släppa',
                'vara',
                'vila',
                'dra',
            ],
        ),
        (
            'tran_2019_03_test.txt',
            [
                'finna',
                'kunna',
                'hitta',
                'gå',
                'vara',
                'kväva',
                'sammanskruvade',
                'vara',
                'ända',
                'vidröra',
                'ruva',
                'växa',
                'öppna',
                'vara',
                'ligga',
                'måste',
                'leva',
                'kunna',
                'ge',
                'finna',
                'öppna',
                'vara',
                'hålla',
                'vara',
                'minna',
                'glömma',
                'anteckna',
                'glömma',
                'sorla',
                'vara',
                'dö',
                'flytta',
                'upphöra',
                'stå',
                'bli',
                'vara',
                'ha',
                'vara',
                'gå',
                'dyka',
                'gå',
                'tränga',
                'stiga',
                'glesna',
                'ljusna',
                'bli',
                'smyga',
                'vara',
                'sitta',
                'ligga',
                'hopvecklade',
            ],
        ),
        ('tran_2020_01_test.txt', ['rulla', 'tugga', 'frusta', 'hölja', 'pejla', 'stanna', 'bli']),
        (
            'tran_2020_02_test.txt',
            ['sparka', 'klänga', 'lövas', 'fylla', 'segla', 'stå', 'komma', 'dra', 'vila', 'sjunka', 'stå'],
        ),
    ]

    corpus = SparvTokenizedCsvCorpus(
        SPARV_ZIPPED_CSV_EXPORT_FILENAME,
        extract_tokens_opts=ExtractTaggedTokensOpts(pos_includes='|VB|', lemmatize=True),
        reader_opts=TextReaderOpts(),
        chunk_size=None,
        tokens_transform_opts=TokensTransformOpts(
            to_lower=True,
        ),
    )

    for i, (filename, tokens) in enumerate(corpus):

        assert filename == expected[i][0]
        assert tokens == expected[i][1]


# def test_count_words():
#     corpus = SparvTokenizedCsvCorpus(
#         '/data/westac/riksdagens-protokoll.1920-2019.sparv4.csv.zip',
#         extract_tokens_opts=ExtractTaggedTokensOpts(lemmatize=False, pos_excludes='|MAD|MID|PAD|'),
#         reader_opts=TextReaderOpts(),
#         chunk_size=None,
#         tokens_transform_opts=TokensTransformOpts(
#             to_lower=True,
#             min_len=1
#         ),
#     )

#     n_counts = {}
#     for filename, tokens in tqdm.tqdm(corpus, total=len(corpus.documents)):
#         n_counts[filename] = len(tokens)

#     n_counts = pd.DataFrame(data=n_counts).set_index('filename', drop=False)

#     df = corpus.documents.merge(n_counts, left_on='filename', right_index=True)

