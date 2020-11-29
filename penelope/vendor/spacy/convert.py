import json
import zipfile
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Any, Callable, Dict, Iterable, Iterator, List, Union

import numpy as np
import pandas as pd
import spacy
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts
from penelope.corpus.readers import SpacyExtractTokensOpts
from spacy.language import Language
from spacy.tokens import Doc

from ._utils import to_text
from .interfaces import ContentType, DocumentPayload, PipelineError

SerializableContent = Union[str, Iterable[str], pd.core.api.DataFrame]


@dataclass
class ContentSerializer:

    serialize: Callable[[SerializableContent], str] = None
    deserialize: Callable[[str], SerializableContent] = None

    @staticmethod
    def identity(content: Any):
        return content

    @staticmethod
    def token_to_text(content: Iterable[str]) -> str:
        return ' '.join(content)

    @staticmethod
    def text_to_token(content: str) -> Iterable[str]:
        return content.split(' ')

    @staticmethod
    def df_to_text(content: pd.DataFrame) -> str:
        return content.to_csv(sep='\t', header=True)

    @staticmethod
    def text_to_df(content: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(content), sep='\t', index_col=0)

    @staticmethod
    def read_text(zf: zipfile.ZipFile, filename: str) -> str:
        return zf.read(filename).decode('utf-8')

    @staticmethod
    def read_binary(zf: zipfile.ZipFile, filename: str) -> bytes:
        return zf.read(filename)

    @staticmethod
    def read_json(zf: zipfile.ZipFile, filename: str) -> Dict:
        return json.loads(ContentSerializer.read_text(zf, filename))

    @staticmethod
    def read_dataframe(zf: zipfile.ZipFile, filename: str) -> pd.DataFrame:
        return pd.read_csv(StringIO(ContentSerializer.read_text(zf, filename)), sep='\t', index_col=0)


CHECKPOINT_SERIALIZERS = {
    ContentType.TEXT: ContentSerializer(serialize=ContentSerializer.identity, deserialize=ContentSerializer.identity),
    ContentType.TOKENS: ContentSerializer(
        serialize=ContentSerializer.token_to_text, deserialize=ContentSerializer.text_to_token
    ),
    ContentType.DATAFRAME: ContentSerializer(
        serialize=ContentSerializer.df_to_text, deserialize=ContentSerializer.text_to_df
    ),
    # FIXME: ADD SPARV XML with as_binary
}


@dataclass
class ContentSerializeOpts:
    content_type_code: int = 0
    document_index_name: str = field(default="document_index.csv")
    as_binary: bool = False

    @property
    def content_type(self) -> ContentType:
        return ContentType(self.content_type_code)

    @content_type.setter
    def content_type(self, value: ContentType):
        self.content_type_code = int(value)


@dataclass
class CheckpointData:
    content_type: ContentType = ContentType.NONE
    document_index: pd.DataFrame = None
    payload_stream: Iterable[DocumentPayload] = None
    serialize_opts: ContentSerializeOpts = None


def store_checkpoint(
    *,
    options: ContentSerializeOpts,
    target_filename: str,
    document_index: pd.DataFrame,
    payload_stream: Iterator[DocumentPayload],
) -> Iterable[DocumentPayload]:

    serializer = CHECKPOINT_SERIALIZERS[options.content_type]

    with zipfile.ZipFile(target_filename, mode="w", compresslevel=zipfile.ZIP_DEFLATED) as zf:

        zf.writestr("options.json", json.dumps(asdict(options)).encode('utf8'))

        if document_index is not None:
            zf.writestr(options.document_index_name, data=document_index.to_csv(sep='\t', header=True))

        for payload in payload_stream:
            zf.writestr(payload.filename, data=serializer.serialize(payload.content))
            yield payload


def load_checkpoint(
    source_filename: str,
) -> CheckpointData:

    with zipfile.ZipFile(source_filename, mode="r") as zf:

        filenames = zf.namelist()

        if "options.json" not in filenames:
            raise PipelineError("Checkpoint file is not valid (has no options.json")

        serialize_opts = ContentSerializeOpts(**ContentSerializer.read_json(zf, "options.json"))

        document_index = None
        if serialize_opts.document_index_name in filenames:
            document_index = ContentSerializer.read_dataframe(zf, serialize_opts.document_index_name)
            filenames.remove(serialize_opts.document_index_name)

        filenames.remove("options.json")

    content_reader = ContentSerializer.read_binary if serialize_opts.as_binary else ContentSerializer.read_text
    serializer = CHECKPOINT_SERIALIZERS[serialize_opts.content_type]

    def payload_stream():
        with zipfile.ZipFile(source_filename, mode="r") as zf:
            for filename in filenames:
                yield DocumentPayload(
                    content_type=serialize_opts.content_type,
                    content=serializer.deserialize(content_reader(zf, filename)),
                    filename=filename,
                )

    data: CheckpointData = CheckpointData(
        content_type=serialize_opts.content_type,
        payload_stream=payload_stream(),
        document_index=document_index,
        serialize_opts=serialize_opts,
    )

    return data


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


def to_vectorized_corpus(
    stream: Iterable[DocumentPayload], vectorize_opts: VectorizeOpts, document_index: pd.DataFrame
) -> VectorizedCorpus:
    vectorizer = CorpusVectorizer()
    terms = (to_text(payload.content) for payload in stream)
    corpus = vectorizer.fit_transform_(terms, documents=document_index, vectorize_opts=vectorize_opts)
    return corpus
