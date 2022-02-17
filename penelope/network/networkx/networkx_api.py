# type: ignore
# flake8: noqa

from __future__ import annotations

from penelope import utility as pu
from penelope.utility.utils import DummyClass


class nx(pu.DummyClass):
    class Graph(DummyClass):
        add_nodes_from = pu.DummyFunction
        add_edges_from = pu.DummyFunction
        get_edge_attributes = pu.DummyFunction
        pydot_layout = pu.DummyFunction

    class nx_pydot(pu.DummyClass):
        pydot_layout = pu.DummyClass

    spring_layout = pu.DummyFunction
    shell_layout = pu.DummyFunction
    circular_layout = pu.DummyFunction
    spectral_layout = pu.DummyFunction


try:
    import networkx as nx  # pylint: disable=unused-import
except ImportError:
    ...
