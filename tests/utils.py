import os
from typing import Callable

import penelope.corpus.readers.text_tokenizer as text_tokenizer

TEST_CORPUS_FILENAME = './penelope/tests/test_data/test_corpus.zip'

# pylint: disable=too-many-arguments

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)

def create_text_tokenizer(
    source_path=TEST_CORPUS_FILENAME,
    transforms=None,
    chunk_size: int=None,
    filename_pattern: str="*.txt",
    filename_filter: str=None,
    fix_whitespaces=False,
    fix_hyphenation=True,
    as_binary: bool=False,
    tokenize: Callable=None,
    filename_fields=None
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
        filename_fields=filename_fields
    )
    reader = text_tokenizer.TextTokenizer(source_path, **kwargs)
    return reader
