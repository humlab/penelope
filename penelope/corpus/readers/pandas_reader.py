import pandas as pd

from penelope.vendor.nltk import word_tokenize

from ..transforms import KnownTransformType
from . import tng


class PandasCorpusReader(tng.CorpusReader):
    def __init__(self, data: pd.DataFrame, text_column='txt', **column_filters):
        text_transformer = tng.TextTransformer(
            transform_opts=tng.TextTransformOpts()
            .clear()
            .add(
                [
                    KnownTransformType.fix_unicode,
                    KnownTransformType.fix_whitespaces,
                    KnownTransformType.fix_hyphenation,
                ]
            )
        )
        super().__init__(
            source=tng.PandasSource(data, text_column=text_column, filename_column='filename', **column_filters),
            transformer=text_transformer,
            tokenizer=word_tokenize,
        )
