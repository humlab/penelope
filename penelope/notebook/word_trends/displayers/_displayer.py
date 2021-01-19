import abc
from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import ipywidgets
from penelope.common.curve_fit import pchip_spline  # , rolling_average_smoother
from penelope.corpus.dtm import VectorizedCorpus

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
    def compile(self, *, corpus: VectorizedCorpus, indices: Sequence[int], **_) -> Any:
        return None

    @abc.abstractmethod
    def plot(self, plot_data: dict, **_):  # pylint: disable=unused-argument
        return

    def clear(self):
        self.output.clear_output()

    def display(self, *, corpus: VectorizedCorpus, indices: Sequence[int], smooth: bool):

        if len(indices) == 0:
            raise ValueError("Nothing to plot!")

        self.clear()
        with self.output:
            plot_data = self.compile(corpus=corpus, indices=indices, smoothers=DEFAULT_SMOOTHERS if smooth else [])
            self.plot(plot_data=plot_data)
