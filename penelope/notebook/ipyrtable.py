# from typing import Union

# import ipyregulartable as irt
# import pandas as pd


# def display_table(data: Union[dict, pd.DataFrame]) -> irt.RegularTableWidget:

#     if isinstance(data, dict):
#         df = pd.DataFrame(data=data)  # .set_index('year')
#     elif isinstance(data, pd.DataFrame):
#         df = data
#     else:
#         raise ValueError(f"Data must be dict or pandas.DataFrame not {type(data)}")

#     w = irt.RegularTableWidget(df)
#     return w
