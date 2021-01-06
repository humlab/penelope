import pandas as pd
from penelope.vendor.nltk import word_tokenize

from . import tng
from .text_transformer import TextTransformer


class DataFrameTextTokenizer(tng.CorpusReader):
    def __init__(self, data: pd.DataFrame, text_column='txt', **column_filters):
        super().__init__(
            source=tng.PandasSource(data, text_column=text_column, filename_column='filename', **column_filters),
            transformer=TextTransformer().fix_unicode().fix_whitespaces().fix_hyphenation(),
            tokenizer=word_tokenize,
        )
