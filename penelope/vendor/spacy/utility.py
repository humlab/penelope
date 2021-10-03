import os
import pathlib
import shutil
import tarfile
from os.path import join as jj
from typing import Union

import requests
from loguru import logger
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


def download_model(
    *,
    lang: str = 'en',
    version: str = '2.3.1',
    model_size: str = 'sm',
    model_type: str = 'core',
    folder: str = '/tmp',
) -> str:

    model_name: str = f'{lang}_{model_type}_web_{model_size}-{version}'

    # https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz'
    if not os.path.isdir(jj(folder, model_name)):

        logger.info(f"Downloading spaCy model: {model_name}")

        url: str = f'https://github.com/explosion/spacy-models/releases/download/{model_name}/{model_name}.tar.gz'

        r = requests.get(url, allow_redirects=True)

        os.makedirs(folder, exist_ok=True)

        filename: str = jj(folder, f'{model_name}.tar.gz')
        with open(filename, 'wb') as fp:
            fp.write(r.content)

        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=folder)

        """Move, and only keep, actual model directory"""
        shutil.move(jj(folder, model_name), jj(folder, f'{model_name}.tmp'))
        shutil.move(
            jj(folder, f'{model_name}.tmp', f'{lang}_{model_type}_web_{model_size}', model_name),
            jj(folder, f'{model_name}'),
        )

        shutil.rmtree(jj(folder, f'{model_name}.tmp'), ignore_errors=True)
        os.remove(filename)

    return jj(folder, model_name)
