from typing import Iterable, List, Union

import numpy as np
import pandas as pd
import spacy
from penelope.corpus.readers import SpacyExtractTokensOpts
from spacy.language import Language
from spacy.tokens import Doc


def spacy_doc_to_annotated_dataframe(spacy_doc: Doc, attributes: List[str]) -> pd.DataFrame:
    """Returns token attribute values from a spacy doc a returns a data frame with given attributes as columns"""
    df = pd.DataFrame(
        data=[tuple(getattr(token, x, None) for x in attributes) for token in spacy_doc],
        columns=attributes,
    )
    return df


def text_to_annotated_dataframe(
    document: str,
    attributes: List[str],
    nlp: Language,
) -> pd.DataFrame:
    """Loads a single text into a spacy doc and returns a data frame with given token attributes columns"""
    return spacy_doc_to_annotated_dataframe(nlp(document), attributes=attributes)


def texts_to_annotated_dataframes(
    documents: Iterable[str],
    attributes: List[str],
    language: Union[Language, str] = "en_core_web_sm",
) -> Iterable[pd.DataFrame]:
    """[summary]

    Parameters
    ----------
    documents : Iterable[str]
        A sequence of text documents
    attributes : List[str]
        A list of spaCy Token properties. Each property will be a column in the returned data frame.
        See https://spacy.io/api/token#attributes for valid properties.
        Example:
        "i", "text", "lemma(_)", "pos(_)", "tag(_)", "dep(_)", "shape",
            "is_alpha", "is_stop", "is_punct", "is_space", "is_digit"


    language : Union[Language, str], optional
        spaCy.Language or a string that specifies the language, by default "en_core_web_sm"

    Returns
    -------
    pd.DataFrame
        A data frame with columns corresponding to each given attribute

    Yields
    -------
    Iterator[pd.DataFrame]
        Seqence of documents represented as data frames
    """
    # TODO: Check taht propertys exists
    # TODO: fetch class property functions? (for performance)

    nlp: Language = spacy.load(language, disable=_get_disables(attributes)) if isinstance(language, str) else language

    for document in documents:
        yield text_to_annotated_dataframe(document, attributes, nlp)


TARGET_MAP = {"lemma": "lemma_", "pos_": "pos_", "ent": "ent_"}


def dataframe_to_tokens(doc: pd.DataFrame, extract_opts: SpacyExtractTokensOpts) -> Iterable[str]:

    if extract_opts.lemmatize is None and extract_opts.target_override is None:
        raise ValueError("a valid target not supplied (no lemmatize or target")

    if extract_opts.target_override:
        target = TARGET_MAP.get(extract_opts.target_override, extract_opts.target_override)
    else:
        target = "lemma_" if extract_opts.lemmatize else "text"

    if target not in doc.columns:
        raise ValueError(f"{extract_opts.target_override} is not valid target for given document (missing column)")

    mask = np.repeat(True, len(doc.index))

    if "is_space" in doc.columns:
        if not extract_opts.is_space:
            mask &= ~(doc.is_space)

    if "is_punct" in doc.columns:
        if not extract_opts.is_punct:
            mask &= ~(doc.is_punct)

    if extract_opts.is_alpha is not None:
        if "is_alpha" in doc.columns:
            mask &= doc.is_alpha == extract_opts.is_alpha

    if extract_opts.is_digit is not None:
        if "is_digit" in doc.columns:
            mask &= doc.is_digit == extract_opts.is_digit

    if extract_opts.is_stop is not None:
        if "is_stop" in doc.columns:
            mask &= doc.is_stop == extract_opts.is_stop

    if "pos_" in doc.columns:

        if len(extract_opts.get_pos_includes() or set()) > 0:
            mask &= doc.pos_.isin(extract_opts.get_pos_includes())

        if len(extract_opts.get_pos_excludes() or set()) > 0:
            mask &= ~(doc.pos_.isin(extract_opts.get_pos_excludes()))

    return doc.loc[mask][target].tolist()


def _get_disables(attributes):
    disable = ['vectors', 'textcat']
    if not any('ent' in x for x in attributes):
        disable.append('ner')

    if not any('dep' in x for x in attributes):
        disable.append('parser')
    return disable


# def extract_text_to_vectorized_corpus(
#     source: TextSource,
#     nlp: Language,
#     *,
#     reader_opts: TextReaderOpts,
#     transform_opts: TextTransformOpts,
#     extract_tokens_opts: SpacyExtractTokensOpts,
#     vectorize_opts: VectorizeOpts,
#     document_index: pd.DataFrame = None,
# ) -> VectorizedCorpus:
#     payload = interfaces.PipelinePayload(source=source, document_index=document_index)
#     pipeline = (
#         spacy_pipeline.SpacyPipeline(payload=payload)
#         .load(reader_opts=reader_opts, transform_opts=transform_opts)
#         .text_to_spacy(nlp=nlp)
#         .spacy_to_dataframe(nlp=nlp, attributes=['text', 'lemma_', 'pos_'])
#         .dataframe_to_tokens(extract_tokens_opts=extract_tokens_opts)
#         .tokens_to_text()
#         .to_dtm(vectorize_opts)
#     )

#     corpus = pipeline.resolve()

#     return corpus
