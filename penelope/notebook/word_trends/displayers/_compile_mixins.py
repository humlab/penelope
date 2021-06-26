import itertools
from typing import Any, Callable, List, Sequence

import bokeh
import scipy
from penelope.common.curve_fit import pchip_spline  # , rolling_average_smoother
from penelope.corpus import VectorizedCorpus
from penelope.utility import take

from .utils import PenelopeBugCheck, column_exists_guard

# pylint: disable=unused-argument


DEFAULT_SMOOTHERS = [pchip_spline]  # , rolling_average_smoother('nearest', 3)]


class LinesDataMixin:
    def compile(self, corpus: VectorizedCorpus, indices: List[int], category_name: str = 'category', **kwargs) -> Any:
        """Compile multiline plot data for token ids `indices`, optionally applying `smoothers` functions"""

        column_exists_guard(corpus.document_index, category_name)

        categories = corpus.document_index[category_name]
        bag_term_matrix = corpus.bag_term_matrix

        if not isinstance(bag_term_matrix, scipy.sparse.spmatrix):
            raise PenelopeBugCheck(f"compile_multiline_data expects scipy.sparse.spmatrix, not {type(bag_term_matrix)}")

        # if hasattr(bag_term_matrix, 'todense'):
        #     bag_term_matrix = bag_term_matrix.todense()

        smoothers: List[Callable] = kwargs.get('smoothers', []) or []
        xs_data = []
        ys_data = []
        for j in indices:
            xs_j = categories
            # ys_j = bag_term_matrix[:, j]
            ys_j = bag_term_matrix.getcol(j).A.ravel()
            for smoother in smoothers:
                xs_j, ys_j = smoother(xs_j, ys_j)
            xs_data.append(xs_j)
            ys_data.append(ys_j)

        data = {
            'xs': xs_data,
            'ys': ys_data,
            'label': [corpus.id2token[token_id].upper() for token_id in indices],
            'color': take(len(indices), itertools.cycle(bokeh.palettes.Category10[10])),
        }
        return data


class CategoryDataMixin:
    def compile(self, corpus: VectorizedCorpus, indices: Sequence[int], category_name: str = 'category', **_) -> Any:
        """Extracts trend vectors for tokens ´indices` and returns a dict keyed by token"""

        column_exists_guard(corpus.document_index, category_name)

        categories = corpus.document_index[category_name]

        if len(categories) != corpus.data.shape[0]:
            raise PenelopeBugCheck(
                f"DTM shape {corpus.data.shape} is not compatible with categories {corpus.data.shape}"
            )

        if not isinstance(corpus.bag_term_matrix, scipy.sparse.spmatrix):
            raise PenelopeBugCheck(f"Expected sparse matrix, found {type(corpus.data)}")

        data = {corpus.id2token[token_id]: corpus.bag_term_matrix.getcol(token_id).A.ravel() for token_id in indices}

        data[category_name] = categories

        return data


class TopTokens2MixIn:
    def compile(self, corpus: VectorizedCorpus, indices: Sequence[int], category_name: str = 'category', **_) -> Any:
        """Extracts trend vectors for tokens ´indices` and returns a dict keyed by token"""
        column_exists_guard(corpus.document_index, category_name)
        top_terms = corpus.get_top_terms(category_column=category_name, n_count=10000, kind='token+count')
        return top_terms
