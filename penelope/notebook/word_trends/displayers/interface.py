import abc
from typing import Sequence, TypeVar

import pandas as pd
from ipywidgets import Output

from penelope.common.curve_fit import pchip_spline

T = TypeVar('T', bound='ITrendDisplayer')

DEFAULT_SMOOTHERS = [pchip_spline]  # , rolling_average_smoother('nearest', 3)]


class ITrendDisplayer(abc.ABC):
    def __init__(self, name: str = "noname", **opts):
        self.output: Output = Output()
        self.name: str = name
        self.opts: dict = opts
        self.width: int = opts.get('width', 1000)
        self.height: int = opts.get('height', 800)

    @abc.abstractmethod
    def setup(self):
        return

    @abc.abstractmethod
    def plot(self, *, data: Sequence[pd.DataFrame], temporal_key: str, **_) -> None:  # pylint: disable=unused-argument
        return

    def clear(self):
        self.output.clear_output()
