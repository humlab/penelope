import os
from typing import Callable

import pandas as pd

import penelope.corpus.readers as readers
from penelope.corpus.interfaces import ITokenizedCorpus
from penelope.utility import flatten

OUTPUT_FOLDER = './tests/output'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TEST_CORPUS_FILENAME = './tests/test_data/test_corpus.zip'

# pylint: disable=too-many-arguments

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)


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


def create_text_tokenizer(
    source_path=TEST_CORPUS_FILENAME,
    transforms=None,
    chunk_size: int = None,
    filename_pattern: str = "*.txt",
    filename_filter: str = None,
    fix_whitespaces=False,
    fix_hyphenation=True,
    as_binary: bool = False,
    tokenize: Callable = None,
    filename_fields=None,
):
    kwargs = dict(
        transforms=transforms,
        chunk_size=chunk_size,
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        fix_whitespaces=fix_whitespaces,
        fix_hyphenation=fix_hyphenation,
        as_binary=as_binary,
        tokenize=tokenize,
        filename_fields=filename_fields,
    )
    reader = readers.TextTokenizer(source_path, **kwargs)
    return reader
