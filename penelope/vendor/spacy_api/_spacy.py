from __future__ import annotations

import os
import pathlib
import re
import shutil
import tarfile
from os.path import join as jj
from typing import Callable, Iterable, Literal, Mapping, Union

import requests
from loguru import logger

from penelope import utility as pu

Language = pu.DummyClass
Doc = pu.DummyClass
Token = pu.DummyClass
load = pu.DummyFunction
compile_prefix_regex = pu.create_dummy_function("")
compile_suffix_regex = pu.create_dummy_function("")
Tokenizer = pu.DummyClass
Vocab = pu.DummyClass

try:
    from spacy import __version__ as spacy_version
    from spacy import attrs, load
    from spacy.cli.download import download
    from spacy.language import Language, Vocab
    from spacy.tokenizer import Tokenizer
    from spacy.tokens import Doc, Token
    from spacy.util import compile_prefix_regex, compile_suffix_regex

except ImportError:
    spacy_version = '0.0.0'


def keep_hyphen_tokenizer(nlp: Language) -> Tokenizer:
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(
        nlp.vocab,
        prefix_search=prefix_re.search,
        suffix_search=suffix_re.search,
        infix_finditer=infix_re.finditer,
        token_match=None,
    )


def spacy_data_path(load_env: bool = False) -> str:
    if load_env:
        pu.load_cwd_dotenv()
    return os.environ.get("SPACY_DATA", "")


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

    spacy_data: str = spacy_data_path(True)
    model: str = prepend_path(model, spacy_data)

    return model


def skip_none_values(d: dict) -> dict:
    return {k: v for k, v in d.items() if v}


def remove_whitespace_entities(doc: Doc) -> Doc:
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc


def check_model_version(load_env: bool = True) -> bool:
    major, minor, _ = spacy_version.split('.')
    model_path: str = spacy_data_path(load_env)

    if model_path == "":
        return False

    if model_path.endswith('/'):
        model_path = model_path[:-1]

    if not model_path:
        return False

    model_major, model_minor, _ = os.path.basename(model_path).split('.')

    if major != model_major or minor != model_minor:
        raise ValueError(
            f"spaCy version {major}.{minor} does not match model version {model_major}.{model_minor}\nUse spacy download {model_path} or run script install-script-models.sh"
        )

    return True


def load_model(
    *,
    model: str | Language,
    vocab: Union[Vocab, bool] = True,
    disable: Iterable[str] = None,
    exclude: Iterable[str] = None,
    keep_hyphens: bool = False,
    remove_whitespace_ents: bool = False,
) -> Language:
    check_model_version()

    if remove_whitespace_ents:
        Language.factories['remove_whitespace_entities'] = lambda _nlp, **_cfg: remove_whitespace_entities

    args: dict = skip_none_values(dict(vocab=vocab, disable=disable, exclude=exclude))

    if isinstance(model, Language):
        return model

    if isinstance(model, str):
        try:
            nlp: Language = load(model, **args)
        except OSError:
            logger.info(f"not found: {model}, downloading...")
            download(model)
            nlp: Language = load(model, **args)

            # try:
            #     name: Union[str, Language] = prepend_spacy_path(model)
            #     nlp: Language = load(name, **args)
            # except OSError:
            #     ...
    else:
        raise ValueError("Expected Language or model name")

    if keep_hyphens:
        nlp.tokenizer = keep_hyphen_tokenizer(nlp)

    return nlp


def load_model_by_parts(
    *,
    lang: str = 'en',
    model_type: str = 'core',
    model_source: str = 'web',
    model_size: str = 'sm',
    vocab: Union[Vocab, bool] = True,
    disable: Iterable[str] = None,
    exclude: Iterable[str] = None,
    keep_hyphens: bool = False,
    remove_whitespace_ents: bool = False,
) -> str:
    model: str = f'{lang}_{model_type}_{model_source}_{model_size}'
    return load_model(
        model=model,
        vocab=vocab,
        disable=disable,
        exclude=exclude,
        keep_hyphens=keep_hyphens,
        remove_whitespace_ents=remove_whitespace_ents,
    )


def download_model_by_name(*, model: str):
    download(model)


def download_model(
    *,
    lang: str = 'en',
    model_type: str = 'core',
    model_source: str = 'web',
    model_size: str = 'sm',
    version: str = '2.3.1',
    folder: str = '/tmp',
) -> str:
    model: str = f'{lang}_{model_type}_{model_source}_{model_size}-{version}'

    if not os.path.isdir(jj(folder, model)):
        logger.info(f"Downloading spaCy model: {model}")

        url: str = f'https://github.com/explosion/spacy-models/releases/download/{model}/{model}.tar.gz'

        r = requests.get(url, allow_redirects=True, timeout=600)

        os.makedirs(folder, exist_ok=True)

        filename: str = jj(folder, f'{model}.tar.gz')
        with open(filename, 'wb') as fp:
            fp.write(r.content)

        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=folder)

        """Move, and only keep, actual model directory"""
        shutil.move(jj(folder, model), jj(folder, f'{model}.tmp'))
        shutil.move(
            jj(folder, f'{model}.tmp', f'{lang}_{model_type}_{model_source}_{model_size}', model),
            jj(folder, f'{model}'),
        )

        shutil.rmtree(jj(folder, f'{model}.tmp'), ignore_errors=True)
        os.remove(filename)

    return jj(folder, model)


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
