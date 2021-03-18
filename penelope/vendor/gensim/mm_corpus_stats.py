import logging

import numpy as np
import pandas as pd
import penelope.vendor.nltk as nltk_utility

logger = logging.getLogger(__name__)


class MmCorpusStatisticsService:
    def __init__(self, corpus, dictionary, language):
        self.corpus = corpus
        self.dictionary = dictionary
        self.stopwords = nltk_utility.stopwords.words(language[1])
        _ = dictionary[0]

    def get_total_token_frequencies(self):
        dictionary = self.corpus.dictionary
        freqencies = np.zeros(len(dictionary.id2token))
        for document in self.corpus:
            for i, f in document:
                freqencies[i] += f
        return freqencies

    def get_document_token_frequencies(self) -> pd.DataFrame:
        """
        Returns a DataFrame with per document token frequencies i.e. "melts" doc-term matrix
        """
        data = ((document_id, x[0], x[1]) for document_id, values in enumerate(self.corpus) for x in values)
        df: pd.DataFrame = pd.DataFrame(list(zip(*data)), columns=['document_id', 'token_id', 'count'])
        # FIXME: Possible bug, right_index=True => right_on='document_id'
        df = df.merge(self.corpus.document_names, left_on='document_id', right_index=True)

        return df

    def compute_word_frequencies(self, remove_stopwords: bool) -> pd.DataFrame:
        id2token = self.dictionary.id2token
        term_freqencies = np.zeros(len(id2token))
        for document in self.corpus:
            for i, f in document:
                term_freqencies[i] += f
        stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df: pd.DataFrame = pd.DataFrame(
            {
                'token_id': list(id2token.keys()),
                'token': list(id2token.values()),
                'frequency': term_freqencies,
                'dfs': list(self.dictionary.dfs.values()),
            }
        )
        df['is_stopword'] = df.token.apply(lambda x: x in stopwords)
        if remove_stopwords is True:
            df = df.loc[(not df.is_stopword)]
        df['frequency'] = df.frequency.astype(np.int64)
        df = df[['token_id', 'token', 'frequency', 'dfs', 'is_stopword']].sort_values('frequency', ascending=False)
        return df.set_index('token_id')

    def compute_document_stats(self) -> pd.DataFrame:
        id2token = self.dictionary.id2token
        stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df: pd.DataFrame = pd.DataFrame(
            {
                'document_id': self.corpus.index,
                'document_name': self.corpus.document_names.document_name,
                'treaty_id': self.corpus.document_names.treaty_id,
                'size': [sum(list(zip(*document))[1]) for document in self.corpus],
                'stopwords': [sum([v for (i, v) in document if id2token[i] in stopwords]) for document in self.corpus],
            }
        ).set_index('document_name')
        df[['size', 'stopwords']] = df[['size', 'stopwords']].astype('int')
        return df

    def compute_word_stats(self) -> pd.DataFrame:
        df = self.compute_document_stats()[['size', 'stopwords']]
        df_agg = df.agg(['count', 'mean', 'std', 'min', 'median', 'max', 'sum']).reset_index()
        legend_map = {
            'count': 'Documents',
            'mean': 'Mean words',
            'std': 'Std',
            'min': 'Min',
            'median': 'Median',
            'max': 'Max',
            'sum': 'Sum words',
        }
        df_agg['index'] = df_agg['index'].apply(lambda x: legend_map[x]).astype('str')
        df_agg = df_agg.set_index('index')
        df_agg[df_agg.columns] = df_agg[df_agg.columns].astype('int')
        return df_agg.reset_index()
