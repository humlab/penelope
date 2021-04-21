from numbers import Number
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


def setup_pandas():

    pd.set_option("max_rows", None)
    pd.set_option("max_columns", None)
    pd.set_option('colheader_justify', 'left')
    pd.set_option('max_colwidth', 300)


def set_default_options():

    set_options(max_rows=None, max_columns=None, colheader_justify='left', max_colwidth=300)


def set_options(**kwargs):
    for k, v in kwargs.items():
        pd.set_option(k, v)


def _create_mask(df: pd.DataFrame, name: str, value: Any, sign: bool = True) -> np.ndarray:
    if isinstance(
        value,
        (
            list,
            set,
        ),
    ):
        m = df[name].isin(value)
    elif isinstance(value, tuple):
        m = df[name].between(*value)
    elif isinstance(value, (bool, Number, str)):
        m = df[name].between(*value)
    else:
        m = df[name] == value
    if not sign:
        m = ~m
    return m


def create_mask2(df: pd.DataFrame, masks: Sequence[dict]) -> np.ndarray:

    v = np.repeat(True, len(df.index))
    for m in masks:
        v &= _create_mask(df, **m)
    return v


def create_mask(doc: pd.DataFrame, args: dict) -> np.ndarray:
    """Creates a mask based on key-values in `criterias`

    Each key-value in `criterias` specifies a filter. The filters are combined using boolean `and`.

    Args:
        doc (pd.DataFrame): Data frame to mask
        criterias (dict): Dict with masking criterias

    Raises:
        ValueError: [description]

    Returns:
        np.ndarray: [description]
    """
    mask = np.repeat(True, len(doc.index))

    if doc is None or len(doc) == 0:
        return mask

    for attr_name, attr_value in args.items():

        attr_value_sign = True
        if attr_value is None:
            continue

        if attr_name not in doc.columns:
            # FIXME: Warn if attribute not in colums!
            continue

        if isinstance(attr_value, tuple):
            # if LIST and tuple is passed, then first element indicates if mask should be negated
            if (
                len(attr_value) != 2
                or not isinstance(attr_value[0], bool)
                or not isinstance(attr_value[1], (list, set))
            ):
                raise ValueError(
                    "when tuple is passed: length must be 2 and first element must be boolean and second must be a list"
                )
            attr_value_sign = attr_value[0]
            attr_value = attr_value[1]

        value_serie: pd.Series = doc[attr_name]
        if isinstance(attr_value, bool):
            # if value_serie.isna().sum() > 0:
            #     raise ValueError(f"data error: boolean column {attr_name} contains np.nan")
            mask &= value_serie == attr_value
        elif isinstance(attr_value, (list, set)):
            mask &= value_serie.isin(attr_value) if attr_value_sign else ~value_serie.isin(attr_value)
        else:
            mask &= value_serie == attr_value

    return mask


class PropertyValueMaskingOpts:
    """A simple key-value filter that returns a mask set to True for items that fulfills all criterias"""

    def __init__(self, **kwargs):
        super().__setattr__('data', kwargs or dict())

    def __getitem__(self, key: int):
        return self.data[key]

    def __len__(self):
        return len(self.data)

    def __setattr__(self, k, v):
        self.data[k] = v

    def __getattr__(self, k):
        try:
            return self.data[k]
        except KeyError:
            return None

    @property
    def props(self) -> Dict:
        return self.data

    def mask(self, doc: pd.DataFrame) -> np.ndarray:

        return create_mask(doc, self.data)

    def apply(self, doc: pd.DataFrame) -> pd.DataFrame:
        if len(self.hot_attributes(doc)) == 0:
            return doc
        return doc[self.mask(doc)]

    def hot_attributes(self, doc: pd.DataFrame) -> List[str]:
        """Returns attributes that __might__ filter tagged frame"""
        return [
            (attr_name, attr_value)
            for attr_name, attr_value in self.data.items()
            if attr_name in doc.columns and attr_value is not None
        ]
