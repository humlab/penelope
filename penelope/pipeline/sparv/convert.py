import re
from io import StringIO

import pandas as pd

from penelope.type_alias import TaggedFrame
from penelope.utility import deprecated

from ..checkpoint import CheckpointOpts, CsvContentSerializer

TRANSLATE_TABLE = " :".maketrans({" ": "_", ":": "|"})


def deserialize_lemma_form(tagged_frame: pd.DataFrame, options: CheckpointOpts) -> pd.Series:
    """Extracts first part of baseform (format of baseform is `|lemma|xyz|`

    Note that lemmas are always lower cased
    """
    lemma_column: str = options.lemma_column
    baseform = pd.Series([x.strip('|').replace(' ', '_').replace(":", "|") for x in tagged_frame[lemma_column]])
    # .apply(lambda x: x.strip('|').replace(' ', '_').replace(":", "|"))

    multi_baseform: pd.Series = baseform.str.contains('|', regex=False)

    if multi_baseform.any():
        baseform.update(baseform.loc[multi_baseform].str.split('|', n=1, expand=True)[0])  # .str[0])

    baseform.update(tagged_frame[baseform == ''].token)

    if options.lower_lemma:
        baseform = pd.Series([x.lower() for x in baseform])
    return baseform


COLON_NUMBER_PATTERN = re.compile(r'\:\d+$')


@deprecated
def to_lemma_form(token: str, baseform: str) -> str:
    lemma = token if baseform.strip('|') == '' else baseform.strip('|').split('|')[0]
    lemma = lemma.replace(' ', '_')
    lemma = COLON_NUMBER_PATTERN.sub('', lemma)
    return lemma


class SparvCsvSerializer(CsvContentSerializer):
    def deserialize(self, *, content: str, options: CheckpointOpts) -> TaggedFrame:
        tagged_frame: TaggedFrame = pd.read_csv(
            StringIO(content),
            sep=options.sep,
            quoting=options.quoting,
            index_col=False,
            skiprows=[1],
            keep_default_na=False,
        )  # .fillna('')

        tagged_frame['baseform'] = deserialize_lemma_form(tagged_frame, options)

        return tagged_frame
