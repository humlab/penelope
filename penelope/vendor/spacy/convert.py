import zipfile
from typing import Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import spacy
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import ExtractTokensOpts2
from penelope.utility.filename_utils import replace_extension
from penelope.vendor.spacy import ContentType, DocumentPayload
from spacy.language import Language
from spacy.tokens import Doc

from ._utils import read_data_frame_from_zip, to_text, write_data_frame_to_zip


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


def dataframe_to_tokens(doc: pd.DataFrame, extract_opts: ExtractTokensOpts2) -> Iterable[str]:

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


def to_vectorized_corpus(
    stream: Iterable[DocumentPayload], vectorize_opts: VectorizeOpts, document_index: pd.DataFrame
) -> VectorizedCorpus:
    vectorizer = CorpusVectorizer()
    terms = (to_text(payload.content) for payload in stream)
    corpus = vectorizer.fit_transform_(terms, documents=document_index, vectorize_opts=vectorize_opts)
    return corpus


def store_data_frame_stream(
    *,
    target_filename: str,
    document_index: pd.DataFrame,
    payload_stream: Iterator[DocumentPayload],
    document_index_name="document_index.csv",
):
    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:
        write_data_frame_to_zip(document_index, document_index_name, zf)
        for payload in payload_stream:
            filename = replace_extension(payload.filename, ".csv")
            write_data_frame_to_zip(payload.content, filename, zf)
            yield payload


def load_data_frame_stream(
    *,
    source_filename: str,
    document_index_name: str = "document_index.csv",
) -> Tuple[Iterable[DocumentPayload], Optional[pd.DataFrame]]:

    document_index = None

    with zipfile.ZipFile(source_filename, mode="r") as zf:

        filenames = zf.namelist()
        if document_index_name in filenames:
            document_index = read_data_frame_from_zip(zf, document_index_name)
            filenames.remove(document_index_name)

    def document_stream():
        with zipfile.ZipFile(source_filename, mode="r") as zf:
            for filename in filenames:
                payload = DocumentPayload(
                    content_type=ContentType.DATAFRAME,
                    content=read_data_frame_from_zip(zf, filename),
                    filename=filename,
                )
                yield payload

    return (document_stream(), document_index)


# def load_data_frame_instream(
#     *,
#     payload: PipelinePayload,
#     source_filename: str,
#     document_index_name: str = "document_index.csv",
# ) -> Iterable[DocumentPayload]:
#     payload.source = source_filename
#     with zipfile.ZipFile(source_filename, mode="r") as zf:
#         filenames = zf.namelist()
#         for filename in filenames:
#             df = read_data_frame_from_zip(zf, filename)
#             if filename == document_index_name:
#                 payload.document_index_source = df
#             else:
#                 payload = DocumentPayload(content_type=ContentType.DATAFRAME, content=df, filename=filename)
#                 yield payload
