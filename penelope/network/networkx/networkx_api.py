# type: ignore
# flake8: noqa
# pylint: disable=unused-argument

from __future__ import annotations

from penelope import utility as pu
from penelope.utility.utils import DummyClass


class nx(pu.DummyClass):
    class Graph(DummyClass):
        add_nodes_from = lambda *args, **kwargs: {}
        add_edges_from = lambda *args, **kwargs: {}
        get_edge_attributes = lambda *args, **kwargs: {}
        pydot_layout = lambda *args, **kwargs: {}

    class nx_pydot(pu.DummyClass):
        pydot_layout = pu.DummyClass

    spring_layout = lambda *args, **kwargs: {}
    shell_layout = lambda *args, **kwargs: {}
    circular_layout = lambda *args, **kwargs: {}
    spectral_layout = lambda *args, **kwargs: {}


try:
    import networkx as nx  # pylint: disable=unused-import
except ImportError:
    ...
