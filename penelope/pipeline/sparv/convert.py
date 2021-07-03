import re
from io import StringIO

import pandas as pd
from penelope.utility import deprecated

from ..checkpoint import CheckpointOpts, CsvContentSerializer
from ..tagged_frame import TaggedFrame

TRANSLATE_TABLE = " :".maketrans({" ": "_", ":": "|"})


def deserialize_lemma_form(tagged_frame: pd.DataFrame) -> pd.Series:

    # baseform = tagged_frame['baseform'].str.strip('|').str.replace(' ', '_').str.replace(":", "|")
    baseform = tagged_frame['baseform'].apply(lambda x: x.strip('|').replace(' ', '_').replace(":", "|"))

    # multi_baseform = baseform.loc[baseform.str.contains('|', regex=False)]
    # if len(multi_baseform) > 0:
    #     baseform.update(multi_baseform.str.split('|', n=1, expand=True)[0]) #.str[0])
    baseform.update(
        baseform.loc[baseform.str.contains('|', regex=False)].str.split('|', n=1, expand=True)[0]
    )  # .str[0])

    # colon_in_baseform = baseform.loc[baseform.str.contains(':', regex=False)]
    # if len(colon_in_baseform) > 0:
    #     baseform.update(colon_in_baseform.str.split(':', n=1, expand=True)[0])

    baseform.update(tagged_frame[baseform == ''].token)

    return baseform


@deprecated
def deserialize_lemma_form_using_translate_id_slower(tagged_frame: pd.DataFrame) -> pd.Series:
    baseform = tagged_frame['baseform'].str.strip('|').str.translate(TRANSLATE_TABLE)
    multi_baseform = baseform.loc[baseform.str.contains('|', regex=False)]
    if len(multi_baseform) > 0:
        baseform.update(multi_baseform.str.split('|', n=1, expand=True)[0])  # .str[0])
    baseform.update(tagged_frame[baseform == ''].token)
    return baseform


COLON_NUMBER_PATTERN = re.compile(r'\:\d+$')


@deprecated
def to_lemma_form(token: str, baseform: str) -> str:
    lemma = token if baseform.strip('|') == '' else baseform.strip('|').split('|')[0]
    lemma = lemma.replace(' ', '_')
    lemma = COLON_NUMBER_PATTERN.sub('', lemma)
    return lemma


class SparvCsvSerializer(CsvContentSerializer):
    def deserialize(self, content: str, options: CheckpointOpts) -> TaggedFrame:
        """Extracts first part of baseform (format of baseform is `|lemma|xyz|`"""
        tagged_frame: TaggedFrame = pd.read_csv(
            StringIO(content),
            sep=options.sep,
            quoting=options.quoting,
            index_col=False,
            skiprows=[1],
            keep_default_na=False,
        )  # .fillna('')

        tagged_frame['baseform'] = deserialize_lemma_form(tagged_frame)
        # tagged_frame['baseform'] = tagged_frame.apply(lambda x: to_lemma_form(x['token'], x['baseform']), axis=1)

        return tagged_frame
