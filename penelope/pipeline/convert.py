from typing import Callable, Iterable, List, Union

import numpy as np
import pandas as pd
from penelope.corpus import CorpusVectorizer, VectorizedCorpus, VectorizeOpts, default_tokenizer
from penelope.corpus.readers import ExtractTaggedTokensOpts, TaggedTokensFilterOpts
from penelope.utility.pos_tags import PoS_Tag_Scheme

from .interfaces import ContentType, DocumentPayload, PipelineError


def _payload_tokens(payload: DocumentPayload) -> List[str]:
    if payload.previous_content_type == ContentType.TEXT:
        return (payload.content[0], default_tokenizer(payload.content[1]))
    return payload.content


def to_vectorized_corpus(
    stream: Iterable[DocumentPayload],
    vectorize_opts: VectorizeOpts,
    document_index: Union[Callable[[], pd.DataFrame], pd.DataFrame],
) -> VectorizedCorpus:
    vectorizer = CorpusVectorizer()
    vectorize_opts.already_tokenized = True
    terms = (_payload_tokens(payload) for payload in stream)
    corpus = vectorizer.fit_transform_(
        terms,
        document_index=document_index,
        vectorize_opts=vectorize_opts,
    )
    return corpus


def tagged_frame_to_tokens(
    doc: pd.DataFrame,
    extract_opts: ExtractTaggedTokensOpts,
    filter_opts: TaggedTokensFilterOpts = None,
    text_column: str = 'text',
    lemma_column: str = 'lemma_',
    pos_column: str = 'pos_',
    verbose: bool = True,  # pylint: disable=unused-argument
) -> Iterable[str]:

    # FIXME: #31 Verify that blank LEMMAS are replaced  by TOKEN
    if extract_opts.lemmatize is None and extract_opts.target_override is None:
        raise ValueError("a valid target not supplied (no lemmatize or target")

    if extract_opts.target_override:
        target = extract_opts.target_override
    else:
        target = lemma_column if extract_opts.lemmatize else text_column

    if target not in doc.columns:
        raise ValueError(f"{target} is not valid target for given document (missing column)")

    mask = np.repeat(True, len(doc.index))

    if filter_opts is not None:
        mask &= filter_opts.mask(doc)

    # FIXME: Merge with filter_opts:
    if pos_column in doc.columns:

        if len(extract_opts.get_pos_includes() or set()) > 0:
            mask &= doc[pos_column].isin(extract_opts.get_pos_includes())

        if len(extract_opts.get_pos_excludes() or set()) > 0:
            mask &= ~(doc[pos_column].isin(extract_opts.get_pos_excludes()))

    return doc.loc[mask][target].tolist()


def tagged_frame_to_token_counts(tagged_frame: pd.DataFrame, pos_schema: PoS_Tag_Scheme, pos_column: str) -> dict:
    """Computes word counts (total and per part-of-speech) given tagged_frame"""

    if tagged_frame is None or len(tagged_frame) == 0:
        return {}

    if not isinstance(tagged_frame, pd.DataFrame):
        raise PipelineError(f"Expected tagged dataframe, found {type(tagged_frame)}")

    if not pos_column:
        raise PipelineError("Name of PoS column in tagged frame MUST be specified (pipeline.payload.memory_store)")

    pos_statistics = (
        tagged_frame.merge(
            pos_schema.PD_PoS_tags,
            how='inner',
            left_on=pos_column,
            right_index=True,
        )
        .groupby('tag_group_name')['tag']
        .size()
    )
    n_raw_tokens = pos_statistics[~(pos_statistics.index == 'Delimiter')].sum()
    token_counts = pos_statistics.to_dict()
    token_counts.update(n_raw_tokens=n_raw_tokens)
    return token_counts
