import os
from penelope.corpus.tokens_transformer import TokensTransformOpts

import numpy as np
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.corpus.readers.interfaces import ExtractTaggedTokensOpts
import random
from collections import defaultdict
from typing import Callable, Iterable, List, Mapping, Tuple

import pandas as pd
from penelope.corpus import ITokenizedCorpus, TextTransformOpts, TokenizedCorpus, metadata_to_document_index
from penelope.corpus.readers import InMemoryReader, TextReader, TextReaderOpts, TextTokenizer
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


def generate_token2id(terms: Iterable[str]) -> Mapping[str, int]:
    token2id = defaultdict()
    token2id.default_factory = token2id.__len__
    for tokens in terms:
        for token in tokens:
            _ = token2id[token]
    return dict(token2id)


def very_simple_corpus(data: List[Tuple[str, List[str]]]) -> TokenizedCorpus:

    reader = InMemoryReader(data, reader_opts=TextReaderOpts(filename_fields="year:_:1"))
    corpus = TokenizedCorpus(reader=reader)
    return corpus


def random_corpus(
    n_docs: int = 5, vocabulary: str = 'abcdefg', min_length: int = 4, max_length: int = 10, years: List[int] = None
) -> List[Tuple[str, List[str]]]:
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
    as_binary: bool = False,
    filename_fields=None,
    index_field=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    fix_hyphenation: bool = True,
    fix_whitespaces: bool = False,
) -> TextReader:
    reader_opts = TextReaderOpts(
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        filename_fields=filename_fields,
        index_field=index_field,
        as_binary=as_binary,
    )
    transform_opts = TextTransformOpts(fix_whitespaces=fix_whitespaces, fix_hyphenation=fix_hyphenation)
    reader = TextReader(source=source_path, reader_opts=reader_opts, transform_opts=transform_opts)
    return reader


def create_tokens_reader(
    source_path=TEST_CORPUS_FILENAME,
    as_binary: bool = False,
    filename_fields=None,
    index_field=None,
    filename_filter: str = None,
    filename_pattern: str = "*.txt",
    fix_hyphenation: bool = True,
    fix_whitespaces: bool = False,
    chunk_size: int = None,
    tokenize: Callable = None,
) -> TextTokenizer:
    reader_opts = TextReaderOpts(
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        filename_fields=filename_fields,
        index_field=index_field,
        as_binary=as_binary,
    )
    transform_opts = TextTransformOpts(fix_whitespaces=fix_whitespaces, fix_hyphenation=fix_hyphenation)
    reader = TextTokenizer(
        source_path,
        reader_opts=reader_opts,
        transform_opts=transform_opts,
        tokenize=tokenize,
        chunk_size=chunk_size,
    )
    return reader

def create_smaller_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus = VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return v_corpus


# def create_bigger_vectorized_corpus(
#     corpus_filename: str,
#     output_tag: str = "xyz_nnvb_lemma",
#     output_folder: str = "./tests/output",
#     count_threshold: int = 5,
# ):
#     filename_field = r"year:prot\_(\d{4}).*"
#     count_threshold = 5
#     output_tag = f"{output_tag}_nnvb_lemma"
#     extract_tokens_opts = ExtractTaggedTokensOpts(
#         pos_includes="|NN|PM|UO|PC|VB|",
#         pos_excludes="|MAD|MID|PAD|",
#         passthrough_tokens=[],
#         lemmatize=True,
#         append_pos=False,
#     )
#     tokens_transform_opts = TokensTransformOpts(
#         only_alphabetic=False,
#         only_any_alphanumeric=False,
#         to_lower=True,
#         to_upper=False,
#         min_len=1,
#         max_len=None,
#         remove_accents=False,
#         remove_stopwords=True,
#         stopwords=None,
#         extra_stopwords=["Örn"],
#         language="swedish",
#         keep_numerals=True,
#         keep_symbols=True,
#     )
#     corpus = vectorize_corpus_workflow(
#         corpus_type="sparv4-csv",
#         input_filename=corpus_filename,
#         output_folder=output_folder,
#         output_tag=output_tag,
#         create_subfolder=True,
#         filename_field=filename_field,
#         count_threshold=count_threshold,
#         extract_tokens_opts=extract_tokens_opts,
#         tokens_transform_opts=tokens_transform_opts,
#     )

#     return corpus
