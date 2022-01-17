import abc
from typing import Callable, List, Sequence

import pandas as pd
import scipy
from penelope.common.curve_fit import pchip_spline
from penelope.corpus import VectorizedCorpus

from ...utility import generate_colors
from .utils import PenelopeBugCheck

# pylint: disable=unused-argument


DEFAULT_SMOOTHERS = [pchip_spline]  # , rolling_average_smoother('nearest', 3)]


class ICompileMixIn(abc.ABC):
    def compile(self, corpus: VectorizedCorpus, indices: List[int], category_name: str, **kwargs) -> dict:
        ...


class MultiLineCompileMixIn(ICompileMixIn):
    def compile(self, corpus: VectorizedCorpus, indices: List[int], category_name: str, **kwargs) -> dict:
        """Compile multiline plot data for token ids `indices`, apply `smoothers` functions. Return dict"""

        guard(corpus, category_name)

        categories = corpus.document_index[category_name]
        bag_term_matrix = corpus.bag_term_matrix

        smoothers: List[Callable] = kwargs.get('smoothers', []) or []
        xs_data, ys_data = [], []

        for j in indices:
            xs_j, ys_j = categories, bag_term_matrix.getcol(j).A.ravel()
            for smoother in smoothers:
                xs_j, ys_j = smoother(xs_j, ys_j)
            xs_data.append(xs_j)
            ys_data.append(ys_j)

        labels: List[str] = [corpus.id2token[token_id].upper() for token_id in indices]
        colors: List[str] = generate_colors(n=len(indices), palette_id=20)

        data: dict = dict(xs=xs_data, ys=ys_data, labels=labels, colors=colors)
        return data


class UnstackedTabularCompileMixIn(ICompileMixIn):
    def compile(
        self, corpus: VectorizedCorpus, indices: Sequence[int], category_name: str = 'category', **_
    ) -> pd.DataFrame:
        """Extracts trend vectors for tokens ´indices` and returns a pd.DataFrame."""

        guard(corpus, category_name)

        categories = corpus.document_index[category_name]

        data: pd.DataFrame = pd.DataFrame(
            data={corpus.id2token[token_id]: corpus.bag_term_matrix.getcol(token_id).A.ravel() for token_id in indices}
        ).assign(category_name=categories)
        return data


class TopTokensCompileMixIn(ICompileMixIn):
    def compile(
        self, corpus: VectorizedCorpus, indices: Sequence[int], category_name: str = 'category', **_
    ) -> pd.DataFrame:
        """Extracts trend vectors for tokens ´indices` and returns a dict keyed by token"""

        guard(corpus, category_name)

        top_terms: pd.DataFrame = corpus.get_top_terms(category_column=category_name, n_top=10000, kind='token+count')

        return top_terms


def guard(corpus: VectorizedCorpus, category_name: str) -> None:

    if category_name not in corpus.document_index.columns:
        raise PenelopeBugCheck(
            f"expected '{category_name}' to be in document index. Found:  {', '.join(corpus.document_index.columns)}"
        )

    if len(corpus.document_index) != corpus.data.shape[0]:
        raise PenelopeBugCheck(f"DTM shape {corpus.data.shape} is not compatible with categories {corpus.data.shape}")

    if not isinstance(corpus.bag_term_matrix, scipy.sparse.spmatrix):
        raise PenelopeBugCheck(f"Expected sparse matrix, found {type(corpus.data)}")
