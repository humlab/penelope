import os
from typing import Callable, Iterable, List, Set, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from penelope.corpus import (
    CorpusVectorizer,
    DocumentIndex,
    Token2Id,
    TokensTransformOpts,
    VectorizedCorpus,
    VectorizeOpts,
    default_tokenizer,
)
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN, ExtractTaggedTokensOpts, PhraseSubstitutions
from penelope.utility import PoS_Tag_Scheme, PropertyValueMaskingOpts

from .interfaces import ContentType, DocumentPayload, PipelineError
from .tagged_frame import TaggedFrame


def _payload_tokens(payload: DocumentPayload) -> List[str]:
    if payload.previous_content_type == ContentType.TEXT:
        return (payload.content[0], default_tokenizer(payload.content[1]))
    return payload.content


def to_vectorized_corpus(
    stream: Iterable[DocumentPayload],
    vectorize_opts: VectorizeOpts,
    document_index: Union[Callable[[], DocumentIndex], DocumentIndex],
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


def tagged_frame_to_tokens(  # pylint: disable=too-many-arguments, too-many-statements
    doc: TaggedFrame,
    extract_opts: ExtractTaggedTokensOpts,
    token2id: Token2Id = None,
    filter_opts: PropertyValueMaskingOpts = None,
    text_column: str = 'text',
    lemma_column: str = 'lemma_',
    pos_column: str = 'pos_',
    transform_opts: TokensTransformOpts = None,
) -> Iterable[str]:
    """Extracts tokens from a tagged document represented as a Pandas data frame.

    Args:
        extract_opts (ExtractTaggedTokensOpts): Part-of-speech/lemma extract options (e.g. PoS-filter)
        filter_opts (PropertyValueMaskingOpts, optional): Filter based on boolean flags in tagged frame. Defaults to None.
        text_column (str, optional): Name of text column in data frame. Defaults to 'text'.
        lemma_column (str, optional): Name of `lemma` column in data frame. Defaults to 'lemma_'.
        pos_column (str, optional): Name of PoS column. Defaults to 'pos_'.

    Returns:
        Iterable[str]: Sequence of extracted tokens
    """
    pad: str = "*"
    phrase_pad: str = "(*)"

    if extract_opts.lemmatize is None and extract_opts.target_override is None:
        raise ValueError("a valid target not supplied (no lemmatize or target")

    if pos_column not in doc.columns:
        raise ValueError(f"configuration error: {pos_column} not in document")

    if extract_opts.target_override:
        target = extract_opts.target_override
    else:
        target = lemma_column if extract_opts.lemmatize else text_column

    if target not in doc.columns:
        raise ValueError(f"{target} is not valid target for given document (missing column)")

    passthroughs: Set[str] = extract_opts.get_passthrough_tokens()
    blocks: Set[str] = extract_opts.get_block_tokens().union('')
    pos_paddings: Set[str] = extract_opts.get_pos_paddings()

    if extract_opts.to_lowercase or extract_opts.lemmatize or (transform_opts and transform_opts.to_lower):
        doc[target] = doc[target].str.lower()
        passthroughs = {x.lower() for x in passthroughs}

    # if extract_opts.block_chars:
    #     for char in extract_opts.block_chars:
    #         doc[target] = doc[target].str.replace(char, '', regex=False)

    """ Phrase detection """
    if extract_opts.phrases is not None:
        found_phrases = detect_phrases(doc[target], extract_opts.phrases, ignore_case=extract_opts.to_lowercase)
        if found_phrases:
            doc = merge_phrases(doc, found_phrases, target_column=target, pad=phrase_pad)
            passthroughs = passthroughs.union({'_'.join(x[1]) for x in found_phrases})

    mask = np.repeat(True, len(doc.index))
    if filter_opts is not None:
        mask &= filter_opts.mask(doc)

    if len(extract_opts.get_pos_includes()) > 0:
        """Don't filter if PoS-include is empty - and don't filter out PoS tokens that should be padded"""
        mask &= doc[pos_column].isin(extract_opts.get_pos_includes().union(extract_opts.get_pos_paddings()))

    if len(extract_opts.get_pos_excludes()) > 0:
        mask &= ~(doc[pos_column].isin(extract_opts.get_pos_excludes()))

    # TODO: Merge extract_opts och transform_opts
    if transform_opts:
        mask &= transform_opts.mask(doc[target])

    if len(passthroughs) > 0:
        # TODO: #52 Make passthrough token case-insensative
        mask |= doc[target].isin(passthroughs)

    if len(blocks) > 0:
        mask &= ~doc[target].isin(blocks)

    filtered_data = doc.loc[mask][[target, pos_column]].copy()

    if extract_opts.global_tf_threshold > 1:
        """
        If global_tf_threshold_mask then filter out tokens below threshold
        Otherwise replace token with `GLOBAL_TF_THRESHOLD_MASK_TOKEN`

        Alternativ implementation:
            1. Compress Token2Id (removed low frequency words)
            2. Remove or mask tokens not in compressed token2id
        """
        if token2id is None or token2id.tf is None:
            raise ValueError("Token2Id.TF not avaliable when mask_threshold_count was requested")

        tg = token2id.get
        cg = token2id.tf.get

        filtered_data['token_id'] = filtered_data[target].apply(tg)
        filtered_data['token_count'] = filtered_data.token_id.apply(cg)

        low_frequency_mask = filtered_data.token_count.fillna(0) < extract_opts.global_tf_threshold

        passthrough_ids: Set[int] = set() if not passthroughs else {tg(w) for w in passthroughs}
        if passthrough_ids:
            low_frequency_mask &= ~filtered_data.token_id.isin(passthrough_ids)

        if extract_opts.global_tf_threshold_mask:
            """Mask low frequency terms"""
            # filtered_data.update(filtered_data[low_frequency_mask].assign(**{target: GLOBAL_TF_THRESHOLD_MASK_TOKEN}))
            filtered_data[target] = filtered_data[target].where(~low_frequency_mask, GLOBAL_TF_THRESHOLD_MASK_TOKEN)
        else:
            """Filter out low frequency terms"""
            filtered_data = filtered_data[~low_frequency_mask]

    token_pos_tuples = filtered_data[[target, pos_column]].itertuples(index=False, name=None)

    if len(pos_paddings) > 0:
        # token_pos_tuples = map(
        #     lambda x: (pad, x[1]) if x[1] in pos_paddings and x[0] not in passthroughs else x, token_pos_tuples
        # )
        token_pos_tuples = [
            (pad, x[1]) if x[1] in pos_paddings and x[0] not in passthroughs else x for x in token_pos_tuples
        ]

    if extract_opts.append_pos:
        tokens = [
            pad if x[0] == pad else f"{x[0].replace(' ', '_')}@{x[1]}" for x in token_pos_tuples if x[0] != phrase_pad
        ]
    else:
        tokens = [x[0].replace(' ', '_') for x in token_pos_tuples if x[0] != phrase_pad]

    return tokens


def detect_phrases(
    target_series: pd.core.api.Series,
    phrases: PhraseSubstitutions,
    ignore_case: str = False,
) -> List[Tuple[int, str, int]]:
    """Detects and updates phrases on document `doc`.

    Args:
        phrases (List[List[str]]): [description]
        doc (pd.core.api.DataFrame): [description]
        target (str): [description]
    """

    if phrases is None:
        return []

    if not isinstance(phrases, (list, dict)):
        raise TypeError("phrase must be dict ot list")

    phrases = (
        {'_'.join(phrase): phrase for phrase in phrases}
        if isinstance(phrases, list)
        else {token.replace(' ', ''): phrase for token, phrase in phrases.items()}
    )

    if ignore_case:
        phrases = {key: [x.lower() for x in phrase] for key, phrase in phrases.items()}

    found_phrases = []
    for replace_token, phrase in phrases.items():

        if len(phrase) < 2:
            continue

        for idx in target_series[target_series == phrase[0]].index:

            if (target_series[idx : idx + len(phrase)] == phrase).all():
                found_phrases.append((idx, replace_token, len(phrase)))

    return found_phrases


def merge_phrases(
    doc: pd.DataFrame,
    phrase_positions: List[Tuple[int, List[str]]],
    target_column: str,
    pad: str = "*",
) -> pd.DataFrame:
    """Returns (same) document with found phrases merged into a single token.
    The first word in phrase is replaced by entire phrase, and consequtive words are replaced by `pad`.
    Note that the phrase will have the same PoS tag as the first word."""
    for idx, token, n_count in phrase_positions:
        doc.loc[idx, target_column] = token
        doc.loc[idx + 1 : idx + n_count - 1, target_column] = pad
        # doc.loc[idx+1:len(phrase) + 1, pos_column] = 'MID'
    return doc


def parse_phrases(phrase_file: str, phrases: List[str]):

    try:
        phrase_specification = {}

        if phrases:
            phrases = [p.split() for p in phrases]
            phrase_specification.update({'_'.join(phrase).replace(' ', '_'): phrase for phrase in (phrases or [])})

        if phrase_file:
            """Expect file to be lines with format:
            ...
            replace_string; the phrase to replace
            ...
            """
            if os.path.isfile(phrase_file):
                with open(phrase_file, "r") as fp:
                    data_str: str = fp.read()
            else:
                data_str: str = phrase_file

            phrase_lines: List[str] = [line for line in data_str.splitlines() if line.strip() != ""]

            phrase_specification.update(
                {
                    key.strip().replace(' ', '_'): phrase.strip().split()
                    for key, phrase in [line.split(";") for line in phrase_lines]
                }
            )

        if len(phrase_specification) == 0:
            return None

        return phrase_specification

    except Exception as ex:
        logger.error(ex)
        raise ValueError("failed to decode phrases. please review file and/or arguments") from ex


def tagged_frame_to_token_counts(tagged_frame: TaggedFrame, pos_schema: PoS_Tag_Scheme, pos_column: str) -> dict:
    """Computes word counts (total and per part-of-speech) given tagged_frame"""

    if tagged_frame is None or len(tagged_frame) == 0:
        return {}

    if not isinstance(tagged_frame, TaggedFrame):
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
