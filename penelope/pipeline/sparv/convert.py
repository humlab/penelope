from io import StringIO

import pandas as pd

from ..checkpoint import CorpusSerializeOpts, TaggedFrameContentSerializer


def extract_lemma(token: str, baseform: str) -> str:
    lemma = token if baseform.strip('|') == '' else baseform.strip('|').split('|')[0]
    return lemma


class SparvCsvSerializer(TaggedFrameContentSerializer):
    def deserialize(self, content: str, options: CorpusSerializeOpts) -> pd.DataFrame:
        """Extracts first part of baseform (format of baseform is `|lemma|xyz|`"""
        df: pd.DataFrame = pd.read_csv(
            StringIO(content),
            sep=options.sep,
            quoting=options.quoting,
            index_col=False,
            skiprows=[1],  # XML <text> tag
            # converters={
            # 'baseform': lambda x: '' if (x or '').strip('|') == '' else x.strip('|').split('|')[0]
            # }
        ).fillna('')
        df['baseform'] = df.apply(lambda x: extract_lemma(x['token'], x['baseform']), axis=1)
        return df
