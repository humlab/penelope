from typing import Any, Dict, Iterable, List, Union

import pandas as pd
import spacy
from penelope.vendor.spacy import prepend_spacy_path
from spacy.language import Language
from spacy.tokens import Doc, Token

from ..tagged_frame import TaggedFrame


def filter_tokens_by_attribute_values(spacy_doc: Doc, attribute_value_filters: dict) -> Iterable[Token]:
    """Filters out tokens based on given attribute value (dict[attribute, bool])
        Whitespaces are always removed from the returned result.
    Args:
        spacy_doc (Doc): spaCy doc
        attribute_value_filters (dict): List of attributes (keys) and values (bool)

    Returns:
        Iterable[Token]: Filtered result
    """

    tokens = (t for t in spacy_doc if not t.is_space)

    if attribute_value_filters is None:
        return tokens

    keys = set(attribute_value_filters.keys())

    keys.discard('is_space')

    if 'is_punct' in keys:
        value = attribute_value_filters['is_punct']
        tokens = (t for t in tokens if t.is_punct == value)
        keys.discard('is_punct')

    keys = {k for k in keys if attribute_value_filters[k] is not None}

    if len(keys) > 0:
        tokens = (t for t in tokens if all(getattr(t, attr) == attribute_value_filters[attr] for attr in keys))

    return tokens


def spacy_doc_to_tagged_frame(
    *,
    spacy_doc: Doc,
    attributes: List[str],
    attribute_value_filters: Dict[str, Any],
) -> TaggedFrame:
    """Returns a data frame with given attributes as columns"""
    tokens = filter_tokens_by_attribute_values(spacy_doc, attribute_value_filters)

    df: TaggedFrame = pd.DataFrame(
        data=[tuple(getattr(token, x, None) for x in attributes) for token in tokens],
        columns=attributes,
    )
    return df


def text_to_tagged_frame(
    document: str,
    attributes: List[str],
    attribute_value_filters: Dict[str, Any],
    nlp: Language,
) -> TaggedFrame:
    """Loads a single text into a spacy doc and returns a data frame with given token attributes columns
    Whitespace tokens are removed."""
    return spacy_doc_to_tagged_frame(
        spacy_doc=nlp(document),
        attributes=attributes,
        attribute_value_filters=attribute_value_filters,
    )


def texts_to_tagged_frames(
    stream: Iterable[str],
    attributes: List[str],
    attribute_value_filters: Dict[str, Any],
    language: Union[Language, str] = "en_core_web_md",
) -> Iterable[TaggedFrame]:
    """[summary]

    Parameters
    ----------
    stream : Iterable[str]
        A sequence of text stream
    attributes : List[str]
        A list of spaCy Token properties. Each property will be a column in the returned data frame.
        See https://spacy.io/api/token#attributes for valid properties.
        Example:
        "i", "text", "lemma(_)", "pos(_)", "tag(_)", "dep(_)", "shape",
            "is_alpha", "is_stop", "is_punct", "is_digit"

        Note: whitespaces (is_space is True) are not stored in the data frame


    language : Union[Language, str], optional
        spaCy.Language or a string that specifies the language, by default "en_core_web_md"

    Returns
    -------
    TaggedFrame
        A data frame with columns corresponding to each given attribute

    Yields
    -------
    Iterator[TaggedFrame]
        Seqence of documents represented as data frames
    """

    """Add SPACY_DATA environment variable if defined"""
    name: Union[str, Language] = prepend_spacy_path(language)

    nlp: Language = spacy.load(name, disable=_get_disables(attributes)) if isinstance(language, str) else language

    for document in stream:
        yield text_to_tagged_frame(document, attributes, attribute_value_filters, nlp)


def _get_disables(attributes):
    disable = ['vectors', 'textcat']
    if not any('ent' in x for x in attributes):
        disable.append('ner')

    if not any('dep' in x for x in attributes):
        disable.append('parser')
    return disable
