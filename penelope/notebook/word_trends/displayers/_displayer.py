import abc
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, TypeVar

import bokeh
import ipywidgets
import scipy
from penelope.common.curve_fit import pchip_spline  # , rolling_average_smoother
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.utility import take

T = TypeVar('T', bound='ITrendDisplayer')

DEFAULT_SMOOTHERS = [pchip_spline]  # , rolling_average_smoother('nearest', 3)]


@dataclass
class ITrendDisplayer(abc.ABC):

    output: ipywidgets.Output = None
    name: str = "noname"

    @abc.abstractmethod
    def setup(self):
        return

    @abc.abstractmethod
    def compile(self, *, corpus: VectorizedCorpus, indices: Sequence[int], **_) -> Dict:
        return None

    @abc.abstractmethod
    def plot(self, corpus: VectorizedCorpus, compiled_data: dict, **_):  # pylint: disable=unused-argument
        return

    def clear(self):
        self.output.clear_output()

    def display(self, *, corpus: VectorizedCorpus, indices: Sequence[int], smooth: bool):

        if len(indices) == 0:
            raise ValueError("Nothing to plot!")

        self.output.clear()
        with self.output:
            plot_data = self.compile(corpus=corpus, indices=indices, smoothers=DEFAULT_SMOOTHERS if smooth else [])
            self.plot(corpus, compiled_data=plot_data)


class PenelopeBugCheck(Exception):
    pass


class MultiLineDataMixin:
    def compile(self, corpus: VectorizedCorpus, indices: List[int], **kwargs) -> dict:
        """Compile multiline plot data for token ids `indicies`, optionally applying `smoothers` functions"""
        xs = corpus.xs_years()
        bag_term_matrix = corpus.bag_term_matrix

        if not isinstance(bag_term_matrix, scipy.sparse.spmatrix):
            raise PenelopeBugCheck(f"compile_multiline_data expects scipy.sparse.spmatrix, not {type(bag_term_matrix)}")

        # if hasattr(bag_term_matrix, 'todense'):
        #     bag_term_matrix = bag_term_matrix.todense()

        # FIXME #107 Error when occurs when compiling multiline data
        smoothers: List[Callable] = kwargs.get('smoothers', []) or []
        xs_data = []
        ys_data = []
        for j in indices:
            xs_j = xs
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


class YearTokenDataMixin:
    def compile(self, corpus: VectorizedCorpus, indices: Sequence[int], **_) -> dict:
        """Extracts token's vectors for tokens ´indices` and returns a dict keyed by token"""
        xs = corpus.xs_years()
        if len(xs) != corpus.data.shape[0]:
            raise PenelopeBugCheck(
                f"DTM shape {corpus.data.shape} is not compatible with year range {corpus.year_range()}"
            )
        if not isinstance(corpus.bag_term_matrix, scipy.sparse.spmatrix):
            raise PenelopeBugCheck(f"Expected sparse matrix, found {type(corpus.data)}")
        data = {corpus.id2token[token_id]: corpus.bag_term_matrix.getcol(token_id).A.ravel() for token_id in indices}
        data['year'] = xs
        return data
