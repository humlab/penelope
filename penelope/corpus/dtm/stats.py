from heapq import nlargest
from numbers import Number
from typing import Container, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from penelope.utility import getLogger

from .interface import IVectorizedCorpusProtocol

logger = getLogger("penelope")


class StatsMixIn:
    def get_top_n_words(
        self: IVectorizedCorpusProtocol,
        n: int = 1000,
        indices: Sequence[int] = None,
    ) -> Sequence[Tuple[str, Number]]:
        """Returns the top n words in a subset of the self sorted according to occurrence. """

        sum_of_token_counts: np.ndarray = (self.data if indices is None else self.data[indices, :]).sum(axis=0).A1

        largest_token_indicies = (-sum_of_token_counts).argsort()[:n]

        largest_tokens = [
            (self.id2token[i], sum_of_token_counts[i]) for i in largest_token_indicies if sum_of_token_counts[i] > 0
        ]

        return largest_tokens

    def get_partitioned_top_n_words(
        self: IVectorizedCorpusProtocol,
        *,
        category_column: str = 'category',
        n_count: int = 100,
        pad: str = None,
        keep_empty: bool = False,
    ) -> dict:
        """Returns top `n_count` terms per category (as defined by `category_column`) as a dict.

        The dict is keyed by category value and each value is a list of tuples (token, count)
        sorted in descending order based on token count.

        Args:
            category_column (str, optional): Column in document index that defines categories. Defaults to 'category'.
            n_count (int, optional): Number of words to return per category. Defaults to 100.
            pad (str, optional): If specified, the lists are padded to be of equal length by appending tuples (`pad`, 0)
            keep_empty (bool, optional): If false, then empty categories are removed
        Returns:
            dict:
        """
        categories = sorted(self.document_index[category_column].unique().tolist())
        indicies_groups = {
            category: self.document_index[(self.document_index.category == category)].index for category in categories
        }
        data = {
            str(category): self.get_top_n_words(n=n_count, indices=indicies_groups[category])
            for category in indicies_groups
        }

        if keep_empty is False:
            data = {c: data[c] for c in data if len(data[c]) > 0}

        if pad is not None:
            if (n_max := max(len(data[c]) for c in data)) != min(len(data[c]) for c in data):
                data = {
                    c: data[c] if len(data[c]) == n_max else data[c] + [(pad, 0)] * (n_max - len(data[c])) for c in data
                }

        return data

    def get_top_terms(
        self: IVectorizedCorpusProtocol,
        *,
        category_column: str = 'category',
        n_count: int = 100,
        kind: str = 'token',
    ) -> pd.DataFrame:
        """Returns top terms per category (as defined by `category_column`) as a dict or pandas data frame.
        The returned data is sorted in descending order.

        Args:
            category_column (str, optional): Column in document index that defines categories. Defaults to 'category'.
            n_count (int, optional): Number of words to return per category. Defaults to 100.
            kind (str, optional): Specifies each category column(s), 'token', 'token+count' (two columns) or single 'token/count' column.

        Returns:
            Union[pd.DataFrame, dict]: [description]
        """

        partitioned_top_n_words = self.get_partitioned_top_n_words(
            category_column=category_column,
            n_count=n_count,
            pad='*',
            keep_empty=False,
        )

        categories = sorted(partitioned_top_n_words.keys())
        if kind == 'token/count':
            data = {
                category: [f'{token}/{count}' for token, count in partitioned_top_n_words[category]]
                for category in categories
            }
        else:
            data = {category: [token for token, _ in partitioned_top_n_words[category]] for category in categories}
            if kind == 'token+count':
                data = {
                    **data,
                    **{
                        f'{category}/Count': [count for _, count in partitioned_top_n_words[category]]
                        for category in categories
                    },
                }

        df = pd.DataFrame(data=data)
        df = df[sorted(df.columns.tolist())]
        return df

    def n_global_top_tokens(self: IVectorizedCorpusProtocol, n_top: int) -> Dict[str, int]:
        """Returns `n_top` most frequent words.

        Parameters
        ----------
        n_top : int
            Number of words to return

        Returns
        -------
        Dict[str, int]
            Most frequent words and their counts, subset of dict `token_counter`

        """
        tokens = {w: self.token_counter[w] for w in nlargest(n_top, self.token_counter, key=self.token_counter.get)}
        return tokens

    def stats(self: IVectorizedCorpusProtocol):
        """Returns (and prints) some corpus status
        Returns
        -------
        dict
            Corpus stats
        """
        stats_data = {
            'bags': self.bag_term_matrix.shape[0],
            'vocabulay_size': self.bag_term_matrix.shape[1],
            'sum_over_bags': self.bag_term_matrix.sum(),
            '10_top_tokens': ' '.join(self.n_global_top_tokens(10).keys()),
        }
        for key in stats_data:
            logger.info('   {}: {}'.format(key, stats_data[key]))
        return stats_data

    def to_n_top_dataframe(self: IVectorizedCorpusProtocol, n_top: int) -> pd.DataFrame:
        """Returns BoW as a Pandas dataframe with the `n_top` most common words.

        Parameters
        ----------
        n_top : int
            Number of top words to return.

        Returns
        -------
        DataFrame
            BoW for top `n_top` words
        """
        v_n_corpus = self.slice_by_n_top(n_top)
        data = v_n_corpus.bag_term_matrix.T
        df = pd.DataFrame(
            data=data.todense(),
            index=[v_n_corpus.id2token[i] for i in range(0, n_top)],
            columns=range(0, v_n_corpus.n_docs),
        )
        return df

    # def get_top_n_words(self, n=1000, indices=None):
    #     """Returns the top n words in a subset of the corpus sorted according to occurrence. """
    #     if indices is None:
    #         sum_words = self.bag_term_matrix.sum(axis=0)
    #     else:
    #         sum_words = self.bag_term_matrix[indices, :].sum(axis=0)

    #     id2token = self.id2token
    #     token_ids = sum_words.nonzero()[1]
    #     words_freq = [(id2token[i], sum_words[0, i]) for i in token_ids]

    #     words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    #     return words_freq[:n]

    def pick_n_top_words(
        self: IVectorizedCorpusProtocol,
        words: Container[str],
        n_top: int,
        descending: bool = False,
    ) -> List[str]:
        """Returns the `n_top` globally most frequent word in `tokens`"""
        words = list(words)
        if len(words) < n_top:
            return words
        token_counts = [self.token_counter.get(w, 0) for w in words]
        most_frequent_words = [words[x] for x in np.argsort(token_counts)[-n_top:]]
        if descending:
            most_frequent_words = list(sorted(most_frequent_words, reverse=descending))
        return most_frequent_words
