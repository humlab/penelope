# import IPython.display

# from ....ipyaggrid_utility import display_grid
# from ..compile_mixins import TopTokensCompileMixIn
# from ..interface import ITrendDisplayer
# from penelope.utility import deprecated

# @deprecated
# class TopTokensSimpleDisplayer(TopTokensCompileMixIn, ITrendDisplayer):
#     def __init__(self, name: str = "TopTokens", **opts):
#         super().__init__(name=name, **opts)

#     def setup(self, *_, **__):
#         return

#     def plot(self, *, plot_data: dict, category_name: str, **_):  # pylint: disable=unused-argument
#         with self.output:
#             g = display_grid(plot_data)
#             IPython.display.display(g)
