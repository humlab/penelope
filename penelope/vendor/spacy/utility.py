from __future__ import annotations

import os
import pathlib
import re
import shutil
import tarfile
from os.path import join as jj
from typing import Callable, Iterable, Literal, Mapping, Union

import requests
import spacy
from loguru import logger
from spacy import attrs
from spacy.cli import download
from spacy.language import Language
from spacy.tokens import Doc, Token

SPACY_DATA = os.environ.get("SPACY_DATA", "")


def keep_hyphen_tokenizer(nlp: Language) -> spacy.tokenizer.Tokenizer:
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return spacy.tokenizer.Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=None,
    )


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


def skip_none_values(d: dict) -> dict:
    return {k: v for k, v in d.items() if v}


def remove_whitespace_entities(doc: Doc) -> Doc:
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc


def load_model(
    *,
    name_or_nlp: str | Language,
    vocab: Union[spacy.Vocab, bool] = True,
    disable: Iterable[str] = None,
    exclude: Iterable[str] = None,
    keep_hyphens: bool = False,
    remove_whitespace_ents: bool = False,
) -> Language:

    if remove_whitespace_ents:
        Language.factories['remove_whitespace_entities'] = lambda _nlp, **_cfg: remove_whitespace_entities

    args: dict = skip_none_values(dict(vocab=vocab, disable=disable, exclude=exclude))

    if isinstance(name_or_nlp, Language):
        return name_or_nlp

    if isinstance(name_or_nlp, str):

        try:
            nlp: Language = spacy.load(name_or_nlp, **args)
        except Exception:
            try:
                name: Union[str, Language] = prepend_spacy_path(name_or_nlp)
                nlp: Language = spacy.load(name, **args)
            except OSError:
                model_path: str = download_model_by_name(model_name=name_or_nlp)
                nlp: Language = spacy.load(model_path, **args)

    if keep_hyphens:
        nlp.tokenizer = keep_hyphen_tokenizer(nlp)

    return nlp


def load_model_by_parts(
    *,
    lang: str = 'en',
    model_type: str = 'core',
    model_source: str = 'web',
    model_size: str = 'sm',
    vocab: Union[spacy.Vocab, bool] = True,
    disable: Iterable[str] = None,
    exclude: Iterable[str] = None,
    keep_hyphens: bool = False,
    remove_whitespace_ents: bool = False,
) -> str:
    model_name: str = f'{lang}_{model_type}_{model_source}_{model_size}'
    return load_model(
        name_or_nlp=model_name,
        vocab=vocab,
        disable=disable,
        exclude=exclude,
        keep_hyphens=keep_hyphens,
        remove_whitespace_ents=remove_whitespace_ents,
    )


def download_model_by_name(*, model_name: str) -> str:
    download(model_name)


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


def token_count_by(
    *,
    doc: Doc,
    target: Literal['lemma', 'lower', 'orth', 'text'] = 'lemma',
    weighting: Literal['count', 'freq'] = 'count',
    include: Callable[[Token], bool] = None,
    n_min_count: int = 2,
    as_strings: bool = False,
) -> Mapping[str | int, int | float]:
    """Return frequency count for `target` in `doc`."""
    target_keys = {'lemma': attrs.LEMMA, 'lower': attrs.LOWER, 'orth': attrs.ORTH, 'text': attrs.TEXT}

    default_exclude: Callable[[Token], bool] = lambda x: x.is_stop or x.is_punct or x.is_space
    exclude: Callable[[Token], bool] = (
        default_exclude if include is None else lambda x: x.is_stop or x.is_punct or x.is_space or not include(x)
    )

    target_weights: Mapping[str | int, int | float] = doc.count_by(target_keys[target], exclude=exclude)

    if weighting == 'freq':
        n_tokens: int = len(doc)
        target_weights = {id_: weight / n_tokens for id_, weight in target_weights.items()}

    store = doc.vocab.strings
    if as_strings:
        bow = {store[word_id]: count for word_id, count in target_weights.items() if count >= n_min_count}
    else:
        bow = target_weights

    return bow
