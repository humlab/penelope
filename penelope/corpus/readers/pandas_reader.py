import pandas as pd

from penelope.vendor.nltk import word_tokenize

from .. import transform as tr
from . import tng


class PandasCorpusReader(tng.CorpusReader):
    def __init__(self, data: pd.DataFrame, text_column='txt', **column_filters):
        super().__init__(
            source=tng.PandasSource(data, text_column=text_column, filename_column='filename', **column_filters),
            transform_opts=tr.TextTransformOpts(transforms="normalize-unicode,dehyphen,normalize-whitespace"),
            tokenizer=word_tokenize,
        )
