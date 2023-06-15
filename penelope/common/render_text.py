from __future__ import annotations

import abc
import zipfile
from functools import cached_property
from io import StringIO
from os.path import isfile, splitext
from typing import Literal

import pandas as pd
from jinja2 import Template

from penelope.corpus import TextTransformOpts
from penelope.corpus.transforms import normalize_whitespace
from penelope.utility import path_add_suffix, replace_extension

# pylint: disable=unused-argument


DEFAULT_TEMPLATE: str = """
<b>Document:</b> {{ document_name }} ({{ document_id }}) <br/>
<h3> {{ document_name }} </h3>
<span style="color: blue;line-height:50%;">
{{ text }}
</span>
<b>Token count:</b> {{ num_tokens }} <br/>
"""


class ITextRepository(abc.ABC):
    def get_text(self, document_name: str) -> str:
        ...

    def get(self, document_name: str) -> dict:
        ...

    @property
    def filenames(self) -> list[str]:
        ...


class IRenderService(abc.ABC):
    def render(self, document_info: dict, kind: Literal['text', 'html']) -> dict | str:
        ...


class Loader(abc.ABC):
    def __init__(self, source: str):
        self.source: str = source

    def load(self, document: str) -> str:  # pylint: disable=unused-argument
        return self._load_text(self.source, document)

    @abc.abstractmethod
    def _load_text(self, source: str, document: str) -> str:
        ...

    @property
    def filenames(self) -> list[str]:
        ...


class ZipLoader(Loader):
    def _load_text(self, source: str, document: str) -> str:
        if not isfile(source):
            raise FileNotFoundError(source)
        with zipfile.ZipFile(source, "r") as fp:
            return fp.read(document).decode("utf-8")

    @cached_property
    def filenames(self) -> list[str]:
        with zipfile.ZipFile(self.source, "r") as fp:
            return fp.namelist()


class ZippedTextCorpusLoader(ZipLoader):
    def load(self, document: str) -> str:
        return normalize_whitespace(self._load_text(self.source, replace_extension(document, '.txt')))


class TokenizedCorpusLoader(ZipLoader):
    def load(self, document: str) -> str:
        source: str = path_add_suffix(self.source, '_pos_csv')
        tagged_frame: pd.DataFrame = pd.read_csv(
            StringIO(self._load_text(source, replace_extension(document, '.txt'))),
            sep='\t',
        )
        column: str = self.probe_column_name(tagged_frame)
        return tagged_frame[column].str.cat(sep=' ')

    def probe_column_name(self, tagged_frame):
        return next((c for c in tagged_frame.columns if 'text' in c or 'token' in c), 'text')


class LemmaCorpusLoader(TokenizedCorpusLoader):
    def probe_column_name(self, tagged_frame):
        return next((c for c in tagged_frame.columns if 'lemma' in c), 'lemma')


class TextRepository(ITextRepository):
    def __init__(
        self,
        *,
        source: str | Loader,
        document_index: pd.DataFrame,
        transforms: str = "normalize-whitespace",
    ):
        self.source: Loader = source if isinstance(source, Loader) else ZippedTextCorpusLoader(source)
        self.document_index: pd.DataFrame = document_index

        # self.subst_puncts = re.compile(r'\s([,?.!"%\';:`](?:\s|$))')
        self.document_name2id: dict[str, int] = (
            document_index.reset_index().set_index('document_name')['document_id'].to_dict()
        )

        self.transformer: TextTransformOpts = TextTransformOpts(transforms=transforms or "")

    def get_text(self, document_name: str) -> str:
        """Loads a document from source"""
        try:
            return self.source.load(document_name)
        except KeyError:  # pylint: disable=bare-except
            return f'document not found in archive: {document_name}'

    def _get_info(self, document_name: str) -> dict:
        """Returns a dict with document metadata from document index"""
        document_name = splitext(document_name)[0]
        try:
            return self.document_index.loc[document_name].to_dict()
        except KeyError as ex:
            raise KeyError(f'document not found in index: {document_name}') from ex

    @property
    def filenames(self) -> list[str]:
        """Returns a list of document namesm in source"""
        return self.source.filenames

    def get(self, document_name: str) -> dict:
        """Returns a dict with document metadata from document index including text"""
        try:
            data: dict = self._get_info(document_name)
            text: str = self.source.load(document_name)
            data |= {'text': self.transformer.transform(text)}
            return data
        except KeyError as ex:  # pylint: disable=bare-except
            return {
                "document_id": "n/a",
                "document_name": document_name,
                "text": ex.args[0],
                "num_tokens": "n/a",
            }


class RenderService(IRenderService):
    """Renders a document using a template"""

    def __init__(self, template: str | Template, links_registry: dict[str, callable] = None):
        self._template: dict[str, Template] = self._to_template(template)
        self.links_registry: dict = links_registry or {}

    @property
    def template(self) -> Template:
        return self._template

    @template.setter
    def template(self, value: str | Template):
        self._template = self._to_template(value)

    def _to_template(self, template: str | Template) -> Template:
        if template is None:
            return Template(DEFAULT_TEMPLATE)
        if isinstance(template, Template):
            return template
        try:
            if isfile(template):
                with open(template, "r", encoding="utf-8") as fp:
                    return Template(fp.read())
        except:  # pylint: disable=bare-except
            pass
        return Template(template)

    def render(self, document_info: dict, kind: Literal['text', 'html']) -> dict | str:
        """Returns a document as a dict or a string"""

        if kind == 'html':
            return self._to_html(document_info)

        if kind == 'text':
            return self._to_text(document_info)

        return document_info.get('text', '')

    def _to_text(self, document_info: dict) -> str:
        text: str = normalize_whitespace(document_info['text'])
        return text

    def _to_html(self, document_info: dict) -> str:
        try:
            if self.template is None:
                return document_info.get('text', 'no template configured')
            document_info['links'] = {key: fn(document_info) for key, fn in self.links_registry.items()}
            return self.template.render(document_info)
        except Exception as ex:
            return f"render failed: {ex}"
