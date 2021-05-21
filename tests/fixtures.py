import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy
from penelope.co_occurrence import ContextOpts, CoOccurrenceComputeResult, term_term_matrix_to_co_occurrences
from penelope.co_occurrence.archive.compute import compute_corpus_co_occurrence
from penelope.corpus import (
    ITokenizedCorpus,
    ReiterableTerms,
    Token2Id,
    TokenizedCorpus,
    VectorizedCorpus,
    dtm,
    metadata_to_document_index,
)
from penelope.corpus.readers import TextReaderOpts, tng
from penelope.utility import extract_filenames_metadata, flatten


class SimpleTestCorpus:
    def __init__(self, filename: str, reader_opts: TextReaderOpts):

        filename_fields = reader_opts.filename_fields
        index_field = reader_opts.index_field

        with open(filename, 'r') as f:
            lines = f.readlines()

        self.corpus_data = [
            dict(
                filename=data[0], title=data[1], text=data[2], tokens=[x.lower() for x in data[2].split() if len(x) > 0]
            )
            for data in [line.split(' # ') for line in lines]
        ]
        self.filenames = [x['filename'] for x in self.corpus_data]
        self.iterator = None

        metadata = extract_filenames_metadata(filenames=self.filenames, filename_fields=filename_fields)
        self.document_index: pd.DataFrame = metadata_to_document_index(metadata, document_id_field=index_field)
        self.document_index['title'] = [x['title'] for x in self.corpus_data]

    @property
    def terms(self):
        return ReiterableTerms(self)

    def _create_iterator(self):
        return ((x['filename'], x['tokens']) for x in self.corpus_data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self._create_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise


class TranströmerCorpus(SimpleTestCorpus):
    def __init__(self):
        # tran_2019_02_test.txt
        meta_fields = ["year:_:1", "year_serial_id:_:2"]
        super().__init__('./tests/test_data/tranströmer.txt', TextReaderOpts(filename_fields=meta_fields))


SIMPLE_CORPUS_ABCDE_5DOCS = [
    ('tran_2019_01_test.txt', ['a', 'b', 'c', 'c']),
    ('tran_2019_02_test.txt', ['a', 'a', 'b', 'd']),
    ('tran_2019_03_test.txt', ['a', 'e', 'e', 'b']),
    ('tran_2020_01_test.txt', ['c', 'c', 'd', 'a']),
    ('tran_2020_02_test.txt', ['a', 'b', 'b', 'e']),
]

SIMPLE_CORPUS_ABCDEFG_7DOCS = [
    ('rand_1991_1.txt', ['b', 'd', 'a', 'c', 'e', 'b', 'a', 'd', 'b']),
    ('rand_1992_2.txt', ['b', 'f', 'e', 'e', 'f', 'e', 'a', 'a', 'b']),
    ('rand_1992_3.txt', ['a', 'e', 'f', 'b', 'e', 'a', 'b', 'f']),
    ('rand_1992_4.txt', ['e', 'a', 'a', 'b', 'g', 'f', 'g', 'b', 'c']),
    ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
    ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
    ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
]

SIMPLE_CORPUS_ABCDEFG_3DOCS = [
    ('rand_1991_5.txt', ['c', 'b', 'c', 'e', 'd', 'g', 'a']),
    ('rand_1991_6.txt', ['f', 'b', 'g', 'a', 'a']),
    ('rand_1993_7.txt', ['f', 'c', 'f', 'g']),
]

SAMPLE_WINDOW_STREAM = [
    ['rand_1991_1.txt', 0, ['*', '*', 'b', 'd', 'a']],
    ['rand_1991_1.txt', 1, ['c', 'e', 'b', 'a', 'd']],
    ['rand_1991_1.txt', 2, ['a', 'd', 'b', '*', '*']],
    ['rand_1992_2.txt', 0, ['*', '*', 'b', 'f', 'e']],
    ['rand_1992_2.txt', 1, ['a', 'a', 'b', '*', '*']],
    ['rand_1992_3.txt', 0, ['e', 'f', 'b', 'e', 'a']],
    ['rand_1992_3.txt', 1, ['e', 'a', 'b', 'f', '*']],
    ['rand_1992_4.txt', 0, ['a', 'a', 'b', 'g', 'f']],
    ['rand_1992_4.txt', 1, ['f', 'g', 'b', 'c', '*']],
    ['rand_1991_5.txt', 0, ['*', 'c', 'b', 'c', 'e']],
    ['rand_1991_6.txt', 0, ['*', 'f', 'b', 'g', 'a']],
]


def very_simple_corpus(data: List[Tuple[str, List[str]]]) -> TokenizedCorpus:

    reader = tng.CorpusReader(
        source=tng.InMemorySource(data),
        reader_opts=TextReaderOpts(filename_fields="year:_:1"),
        transformer=None,  # already tokenized
    )
    corpus = TokenizedCorpus(reader=reader)
    return corpus


def random_corpus(
    n_docs: int = 5, vocabulary: str = 'abcdefg', min_length: int = 4, max_length: int = 10, years: List[int] = None
) -> List[Tuple[str, List[str]]]:
    def random_tokens():

        return [random.choice(vocabulary) for _ in range(0, random.choice(range(min_length, max_length)))]

    return [(f'rand_{random.choice(years or [0])}_{i}.txt', random_tokens()) for i in range(1, n_docs + 1)]


def very_simple_term_term_matrix(corpus: ITokenizedCorpus) -> scipy.sparse.spmatrix:

    term_term_matrix: scipy.sparse.spmatrix = (
        dtm.CorpusVectorizer()
        .fit_transform(corpus, already_tokenized=True, vocabulary=corpus.token2id)
        .co_occurrence_matrix()
    )
    return term_term_matrix


def very_simple_co_occurrences(corpus: ITokenizedCorpus) -> pd.DataFrame:

    term_term_matrix: scipy.sparse.spmatrix = very_simple_term_term_matrix(corpus)

    co_occurrences: pd.DataFrame = term_term_matrix_to_co_occurrences(
        term_term_matrix=term_term_matrix,
        threshold_count=1,
        ignore_ids=None,
    )

    return co_occurrences


def very_simple_corpus_co_occurrences(corpus: TokenizedCorpus, context_opts: ContextOpts) -> CoOccurrenceComputeResult:

    value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
        stream=corpus,
        context_opts=context_opts,
        token2id=corpus.token2id,
        document_index=corpus.document_index,
        global_threshold_count=None,
        ingest_tokens=False,
    )

    return value


class MockedProcessedCorpus(ITokenizedCorpus):
    def __init__(self, mock_data):
        self.data = [(f, self.generate_document(ws)) for f, ws in mock_data]
        self.token2id: Token2Id = self.create_token2id()
        self.n_tokens = {f: len(d) for f, d in mock_data}
        self.iterator = None
        self._metadata = [dict(filename=filename, year=filename.split('_')[1]) for filename, _ in self.data]
        self._documents = metadata_to_document_index(self._metadata)

    @property
    def terms(self):
        return [tokens for _, tokens in self.data]

    @property
    def filenames(self) -> List[str]:
        return list(self.document_index.filename)

    @property
    def metadata(self):
        return self._metadata

    @property
    def document_index(self) -> pd.DataFrame:
        return self._documents

    def create_token2id(self) -> Token2Id:
        return Token2Id({w: i for i, w in enumerate(sorted(list(set(flatten([x[1] for x in self.data])))))})

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = ((x, y) for x, y in self.data)
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise

    def generate_document(self, words):
        if isinstance(words, str):
            document = words.split()
        else:
            document = flatten([n * w for n, w in words])
        return document


def create_smaller_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus = VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return v_corpus


# Corpus windows test data:
TRANSTRÖMMER_CORPUS_NNVB_LEMMA = [
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


TRANSTRÖMMER_NNVB_LEMMA_WINDOWS = [
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
