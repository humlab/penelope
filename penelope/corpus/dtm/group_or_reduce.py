from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Protocol, Sequence, TypeVar, Union

import numpy as np
import pandas as pd

from ..document_index import DocumentIndex

if TYPE_CHECKING:
    from .interface import IVectorizedCorpus
    from .vectorized_corpus import VectorizedCorpus

T = TypeVar("T", int, str)


class IVectorizedCorpusProtocol(Protocol):
    @property
    def create(self) -> IVectorizedCorpus:
        ...

    @property
    def document_index(self) -> pd.DataFrame:
        ...


class GroupByYearMixIn:
    def group_by_year(self: IVectorizedCorpusProtocol) -> VectorizedCorpus:
        """Returns a new corpus where documents have been grouped and summed up by year."""

        X = self.bag_term_matrix  # if X is None else X
        df = self.document_index  # if df is None else df

        min_value, max_value = df.year.min(), df.year.max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]))

        for i in range(0, Y.shape[0]):  # pylint: disable=unsubscriptable-object

            indices = list((df.loc[df.year == min_value + i].index))

            if len(indices) > 0:
                Y[i, :] = X[indices, :].sum(axis=0)

        years = list(range(min_value, max_value + 1))
        document_index = pd.DataFrame(
            {'year': years, 'category': years, 'filename': map(str, years), 'document_name': map(str, years)}
        )

        corpus: VectorizedCorpus = self.create(
            Y, token2id=self.token2id, document_index=document_index, word_counts=self.word_counts
        )

        return corpus

    # CONSIDER: Refactor away function (make use of `collapse_by_category`)
    def group_by_year2(self: IVectorizedCorpusProtocol, aggregate_function='sum', dtype=None) -> VectorizedCorpus:
        """Variant of `group_by_year` where aggregate function can be specified."""

        assert aggregate_function in {'sum', 'mean'}

        X = self.bag_term_matrix  # if X is None else X
        df = self.document_index  # if df is None else df

        min_value, max_value = df.year.min(), df.year.max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]), dtype=(dtype or X.dtype))

        for i in range(0, Y.shape[0]):  # pylint: disable=unsubscriptable-object

            indices = list((df.loc[df.year == min_value + i].index))

            if len(indices) > 0:
                if aggregate_function == 'mean':
                    Y[i, :] = X[indices, :].mean(axis=0)
                else:
                    Y[i, :] = X[indices, :].sum(axis=0)

                # Y[i,:] = self._group_aggregate_functions[aggregate_function](X[indices,:], axis=0)

        years = list(range(min_value, max_value + 1))

        document_index = pd.DataFrame({'year': years, 'filename': map(str, years), 'document_name': map(str, years)})

        corpus: IVectorizedCorpus = self.create(
            Y, token2id=self.token2id, document_index=document_index, word_counts=self.word_counts
        )
        return corpus


# class ReduceByCatergoryMixIn:
#     def setup_categories(self, document_index: pd.DataFrame, specifier: Union[str, dict]) -> List[int]:
#         """Returns contineous (document index) category series and (unique) category list for given specifier"""
#         if isinstance(specifier, str):
#             if specifier in ('decade', 'lustrum'):
#                 d = 5 if specifier == 'lustrum' else 10
#                 _category_series: pd.Series = document_index.year.apply(lambda x: x - int(x % d))
#                 _categories = list(range(_category_series.min(), _category_series.max() + 1, d))
#             else:
#                 raise ValueError(f"Category {specifier}")
#         else:
#             _category_series: pd.Series = document_index.year.apply(lambda x: specifier.get(x, None))
#             _categories = list(sorted(_category_series.unique()))

#         return _categories, _category_series

#     def reduce_by_categories(self, category_series: pd.Series, categories: List[int]) -> VectorizedCorpus:
#         """Creates new corpus by reducing (summing) rows having same category"""
#         X = self.bag_term_matrix
#         df = self.document_index
#         df['category'] = category_series
#         Y: np.ndarray = np.zeros((len(categories), X.shape[1]), dtype=X.dtype)
#         for i in range(0, Y.shape[0]):  # pylint: disable=unsubscriptable-object
#             indices = list((df.loc[df.category == categories[i]].index))
#             if len(indices) > 0:
#                 Y[i, :] = X[indices, :].sum(axis=0)

#         _index: DocumentIndex = DocumentIndex(self.document_index).collapse_by_integer_column(
#             category_series, categories
#         )

#         _corpus = VectorizedCorpus(
#             Y,
#             token2id=self.token2id,
#             document_index=_index,
#             word_counts=self.word_counts,
#         )

#         return _corpus

#     def group_by_year_categories(self, category_specifier: Union[str, dict]) -> VectorizedCorpus:
#         """Returns a new corpus where documents have been grouped and summed up by year groups."""

#         if category_specifier == 'year':
#             return self.group_by_year()

#         categories, category_series = self.setup_categories(
#             document_index=self.document_index, specifier=category_specifier
#         )

#         corpus = self.reduce_by_categories(category_series, categories)

#         return corpus


class GroupByCategoryMixIn:
    def group_by_category(
        self: IVectorizedCorpusProtocol, category_column: str, categories: Sequence[T] = None
    ) -> VectorizedCorpus:
        """Groups document index by category column,

        Args:
            category_column (str): The column to group by, must exist, must be of int or str type
            transformer (callable, dict, None): Transforms to apply to column before grouping
            index_values (pd.Series, List[T]): pandas index of returned document index

        Returns
        -------
        corpus: VectorizedCorpus
            DTM of size K where K is the number of unique categorical values in `df[column]`
            INDEX of length K with category values as DOCUMENT_NAME, where i:th value is category of i:th row in returned matrix
        """

        if category_column not in self.document_index:
            pass

        categories: Sequence[T] = categories or self._get_category_column_values(category_column)
        bag_term_matrix = self._reduce_bag_term_matrix(categories=categories, category_column=category_column)
        document_index = self._reduce_document_index(categories=categories, category_column=category_column)

        corpus: VectorizedCorpus = self.create(
            bag_term_matrix,
            token2id=self.token2id,
            document_index=document_index,
            word_counts=self.word_counts,
        )

        return corpus

    def _get_category_column_values(self, category_column: str) -> Sequence[T]:
        categories: List[Any] = list(sorted(self.document_index[category_column].unique().tolist()))
        return categories

    def _reduce_document_index(self, *, category_column: str, categories: List[Union[int, str]]) -> pd.DataFrame:
        _index: DocumentIndex = DocumentIndex(self.document_index).group_by_column(
            column_name=category_column, index_values=categories
        )
        return _index

    def _reduce_bag_term_matrix(self, *, categories: List[Union[int, str]], category_column: str):
        """Creates new corpus by reducing (summing) rows having same document attribute"""

        bag_term_matrix = np.zeros((len(categories), self.bag_term_matrix.shape[1]), dtype=self.bag_term_matrix.dtype)

        for i, value in enumerate(categories):

            indices = list((self.document_index.loc[self.document_index[category_column] == value].index))

            bag_term_matrix[i, :] = self.bag_term_matrix[indices, :].mean(axis=0)

        return bag_term_matrix
