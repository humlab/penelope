from __future__ import annotations

import os
import pathlib
import shutil
import tarfile
from os.path import join as jj
from typing import Union

import requests
import spacy
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


def load_model(*, name_or_nlp: str | Language, disables: str, version: str, folder: str) -> Language:
    if isinstance(name_or_nlp, str):
        try:
            name: Union[str, Language] = prepend_spacy_path(name_or_nlp)
            name_or_nlp = spacy.load(name, disable=disables)
        except OSError:
            model_path: str = download_model_by_name(model_name=name_or_nlp, version=version, folder=folder)
            name_or_nlp = spacy.load(model_path, disable=disables)

    return name_or_nlp


def download_model_by_name(
    *,
    model_name: str = 'sm',
    version: str = '2.3.1',
    folder: str = '/tmp',
) -> str:
    lang, model_type, model_source, model_size = model_name.split('_')
    return download_model(
        lang=lang,
        model_type=model_type,
        model_source=model_source,
        model_size=model_size,
        version=version,
        folder=folder,
    )


def download_model(
    *,
    lang: str = 'en',
    model_type: str = 'core',
    model_source: str = 'web',
    model_size: str = 'sm',
    version: str = '2.3.1',
    folder: str = '/tmp',
) -> str:

    model_name: str = f'{lang}_{model_type}_{model_source}_{model_size}-{version}'

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
            jj(folder, f'{model_name}.tmp', f'{lang}_{model_type}_{model_source}_{model_size}', model_name),
            jj(folder, f'{model_name}'),
        )

        shutil.rmtree(jj(folder, f'{model_name}.tmp'), ignore_errors=True)
        os.remove(filename)

    return jj(folder, model_name)
