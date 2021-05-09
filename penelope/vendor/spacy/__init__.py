import os
import pathlib
from typing import Union

from spacy.language import Language

SPACY_DATA = os.environ.get("SPACY_DATA", "")


def prepend_path(model: Union[Language, str], path: str) -> Union[Language, str]:
    """Prepends `model` with `path` if it is an existing folder"""
    if not isinstance(model, (str, pathlib.Path)):
        return model

    if path == "" or os.path.dirname(model) != '':
        return model

    if not pathlib.Path(path).exists():
        return model

    model: str = os.path.join(path, model)

    return model


def prepend_spacy_path(model: Union[Language, str]) -> Union[Language, str]:
    """Prepends `model` with SPACY_DATA environment if set and model is a string"""
    model: str = prepend_path(model, SPACY_DATA)
    return model
