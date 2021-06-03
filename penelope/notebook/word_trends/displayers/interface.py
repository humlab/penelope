import abc
from typing import Any, Sequence, TypeVar

from ipywidgets import Output
from penelope.common.curve_fit import pchip_spline
from penelope.corpus import VectorizedCorpus

T = TypeVar('T', bound='ITrendDisplayer')

DEFAULT_SMOOTHERS = [pchip_spline]  # , rolling_average_smoother('nearest', 3)]


class ITrendDisplayer(abc.ABC):
    def __init__(self, name: str = "noname"):
        self.output: Output = Output()
        self.name: str = name

    @abc.abstractmethod
    def setup(self):
        return

    @abc.abstractmethod
    def compile(self, *, corpus: VectorizedCorpus, indices: Sequence[int], category_name: str, **_) -> Any:
        return None

    @abc.abstractmethod
    def plot(self, *, plot_data: dict, category_name: str, **_):  # pylint: disable=unused-argument
        return

    def clear(self):
        self.output.clear_output()

    def display(self, *, corpus: VectorizedCorpus, indices: Sequence[int], smooth: bool, category_name: str):

        if len(indices) == 0:
            raise ValueError("Nothing to plot!")

        self.clear()
        with self.output:
            plot_data = self.compile(
                corpus=corpus,
                indices=indices,
                category_name=category_name,
                smoothers=DEFAULT_SMOOTHERS if smooth else [],
            )
            self.plot(plot_data=plot_data, category_name=category_name)
