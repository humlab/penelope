import abc
import itertools
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, TypeVar

import bokeh
import ipywidgets
import pandas as pd
import scipy
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.utility import take


@dataclass
class WordTrendData:
    corpus: VectorizedCorpus = None
    corpus_folder: str = None
    corpus_tag: str = None
    compute_options: Dict = None
    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None


T = TypeVar('T', bound='ITrendDisplayer')


@dataclass
class ITrendDisplayer(abc.ABC):

    output: ipywidgets.Output = None
    data: WordTrendData = None
    name: str = "noname"

    @abc.abstractmethod
    def setup(self):
        return

    @abc.abstractmethod
    def compile(self, *, corpus: VectorizedCorpus, indices: Sequence[int], **_) -> Dict:
        return None

    @abc.abstractmethod
    def plot(self, data: Dict, **_):  # pylint: disable=unused-argument
        return

    def clear(self):
        self.output.clear_output()


class PenelopeBugCheck(Exception):
    pass


class MultiLineDataMixin:
    def compile(self, corpus: VectorizedCorpus, indices: List[int], **kwargs) -> Dict:
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
    def compile(self, corpus: VectorizedCorpus, indices: Sequence[int], **_) -> Dict:
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
