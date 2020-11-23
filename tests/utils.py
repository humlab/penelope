import os
import random
from collections import defaultdict
from typing import Callable

import pandas as pd
from penelope.corpus import ITokenizedCorpus, TextTransformOpts, TokenizedCorpus
from penelope.corpus.readers import InMemoryReader, TextReader, TextTokenizer
from penelope.utility import flatten

OUTPUT_FOLDER = './tests/output'
TEST_DATA_FOLDER = './tests/test_data'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TEST_CORPUS_FILENAME = os.path.join(TEST_DATA_FOLDER, 'test_corpus.zip')
TRANSTRÖMMER_ZIPPED_CSV_EXPORT_FILENAME = os.path.join(TEST_DATA_FOLDER, 'tranströmer_corpus_export.csv.zip')

# pylint: disable=too-many-arguments

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)

# http://www.nltk.org/howto/collocations.html
# PMI


def generate_token2id(terms):
    token2id = defaultdict()
    token2id.default_factory = token2id.__len__
    for tokens in terms:
        for token in tokens:
            _ = token2id[token]
    return dict(token2id)


def very_simple_corpus(documents):

    reader = InMemoryReader(documents, filename_fields="year:_:1")
    corpus = TokenizedCorpus(reader=reader)
    return corpus


def random_corpus(n_docs: int = 5, vocabulary: str = 'abcdefg', min_length=4, max_length=10, years=None):
    def random_tokens():

        return [random.choice(vocabulary) for _ in range(0, random.choice(range(min_length, max_length)))]

    return [(f'rand_{random.choice(years or [0])}_{i}.txt', random_tokens()) for i in range(1, n_docs + 1)]


class MockedProcessedCorpus(ITokenizedCorpus):
    def __init__(self, mock_data):
        self.data = [(f, self.generate_document(ws)) for f, ws in mock_data]
        self.token2id = self.create_token2id()
        self.n_tokens = {f: len(d) for f, d in mock_data}
        self.iterator = None
        self._metadata = [dict(filename=filename, year=filename.split('_')[1]) for filename, _ in self.data]
        self._documents = pd.DataFrame(self._metadata)

    @property
    def terms(self):
        return [tokens for _, tokens in self.data]

    @property
    def filenames(self):
        return list(self.documents.filename)

    @property
    def metadata(self):
        return self._metadata

    @property
    def documents(self):
        return self._documents

    def create_token2id(self):
        return {w: i for i, w in enumerate(sorted(list(set(flatten([x[1] for x in self.data])))))}

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


def create_text_reader(
    source_path=TEST_CORPUS_FILENAME,
    transforms=None,
    as_binary: bool = False,
    filename_fields=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    fix_hyphenation=True,
    fix_whitespaces=False,
):
    kwargs = dict(
        transforms=transforms,
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        filename_fields=filename_fields,
        as_binary=as_binary,
        text_transform_opts=TextTransformOpts(fix_whitespaces=fix_whitespaces, fix_hyphenation=fix_hyphenation),
    )
    reader = TextReader(source=source_path, **kwargs)
    return reader


def create_text_tokenizer(
    source_path=TEST_CORPUS_FILENAME,
    as_binary: bool = False,
    filename_fields=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    fix_hyphenation=True,
    fix_whitespaces=False,
    transforms=None,
    chunk_size: int = None,
    tokenize: Callable = None,
):
    kwargs = dict(
        transforms=transforms,
        chunk_size=chunk_size,
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        filename_fields=filename_fields,
        as_binary=as_binary,
        tokenize=tokenize,
        text_transform_opts=TextTransformOpts(fix_whitespaces=fix_whitespaces, fix_hyphenation=fix_hyphenation),
    )
    reader = TextTokenizer(source_path, **kwargs)
    return reader
