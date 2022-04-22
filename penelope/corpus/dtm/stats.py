from numbers import Number
from typing import Container, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from .interface import IVectorizedCorpusProtocol

# pylint: disable=no-member


class StatsMixIn:
    def get_top_n_words(
        self: IVectorizedCorpusProtocol,
        n: int = 1000,
        indices: Sequence[int] = None,
    ) -> Sequence[Tuple[str, Number]]:
        """Returns the top n words in a subset of the self sorted according to occurrence."""

        sum_of_token_counts: np.ndarray = (self.data if indices is None else self.data[indices, :]).sum(axis=0).A1

        largest_token_indices = (-sum_of_token_counts).argsort()[:n]

        largest_tokens = [
            (self.id2token[i], sum_of_token_counts[i]) for i in largest_token_indices if sum_of_token_counts[i] > 0
        ]

        return largest_tokens

    def nlargest(
        self: IVectorizedCorpusProtocol, n_top: int, *, sort_indices: bool = False, override: bool = False
    ) -> np.ndarray:
        """Return indices for the `n_top` most frequent terms in DTM
        Note: indices are sorted by TF count as default."""
        n_top = min(n_top, len(self.term_frequency))
        indices: np.ndarray = np.argpartition(self.term_frequency if not override else self.term_frequency0, -n_top)[
            -n_top:
        ]
        if sort_indices:
            indices.sort()
        return indices

    def get_partitioned_top_n_words(
        self: IVectorizedCorpusProtocol,
        *,
        category_column: str = 'category',
        n_top: int = 100,
        pad: str = None,
        keep_empty: bool = False,
    ) -> dict:
        """Returns top `n_top` terms per category (as defined by `category_column`) as a dict.

        The dict is keyed by category value and each value is a list of tuples (token, count)
        sorted in descending order based on token count.

        Args:
            category_column (str, optional): Column in document index that defines categories. Defaults to 'category'.
            n_top (int, optional): Number of words to return per category. Defaults to 100.
            pad (str, optional): If specified, the lists are padded to be of equal length by appending tuples (`pad`, 0)
            keep_empty (bool, optional): If false, then empty categories are removed
        Returns:
            dict:
        """
        categories = sorted(self.document_index[category_column].unique().tolist())
        indices_groups = {
            category: self.document_index[(self.document_index[category_column] == category)].index
            for category in categories
        }
        data = {
            str(category): self.get_top_n_words(n=n_top, indices=indices_groups[category])
            for category in indices_groups
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
        n_top: int = 100,
        kind: str = 'token',
    ) -> pd.DataFrame:
        """Returns top terms per category (as defined by `category_column`) as a dict or pandas data frame.
        The returned data is sorted in descending order.

        Args:
            category_column (str, optional): Column in document index that defines categories. Defaults to 'category'.
            n_top (int, optional): Number of words to return per category. Defaults to 100.
            kind (str, optional): Specifies each category column(s), 'token', 'token+count' (two columns) or single 'token/count' column.

        Returns:
            Union[pd.DataFrame, dict]: [description]
        """

        partitioned_top_n_words = self.get_partitioned_top_n_words(
            category_column=category_column, n_top=n_top, pad='*', keep_empty=False
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

    def pick_top_tf_map(self: IVectorizedCorpusProtocol, n_top: int) -> Dict[str, int]:
        """Returns `n_top` largest tokens and TF as a dict"""
        tokens = {self.id2token[i]: self.term_frequency[i] for i in self.nlargest(n_top)}
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
            '10_top_tokens': ' '.join(self.pick_top_tf_map(10).keys()),
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

    def pick_n_top_words(
        self: IVectorizedCorpusProtocol,
        words: Container[str],
        n_top: int,
        descending: bool = False,
    ) -> List[str]:
        """Returns the `n_top` globally most frequent word in `tokens`"""
        words = list(words)
        n_top = n_top or len(words)
        if len(words) < n_top:
            return words
        # FIXME: What to do if overriden term frequency?
        fg = self.token2id.get
        tf = self.term_frequency
        token_counts = [tf[fg(w)] for w in words]
        most_frequent_words = [words[x] for x in np.argsort(token_counts)[-n_top:]]
        if descending:
            most_frequent_words = list(sorted(most_frequent_words, reverse=descending))
        return most_frequent_words
