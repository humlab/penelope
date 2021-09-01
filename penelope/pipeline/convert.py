from typing import Callable, Iterable, List, Set, Union

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
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN, ExtractTaggedTokensOpts
from penelope.utility import PoS_Tag_Scheme, PropertyValueMaskingOpts

from .interfaces import ContentType, DocumentPayload
from .phrases import PHRASE_PAD, detect_phrases, merge_phrases


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


# def filter_tagged_frame(  # pylint: disable=too-many-arguments, too-many-statements
#     tagged_frame: pd.DataFrame,
#     extract_opts: ExtractTaggedTokensOpts,
#     token2id: Token2Id = None,
#     filter_opts: PropertyValueMaskingOpts = None,
#     text_column: str = 'text',
#     lemma_column: str = 'lemma_',
#     pos_column: str = 'pos_',
#     transform_opts: TokensTransformOpts = None,
# ) -> Iterable[str]:
#     """Filter tagged frame `doc` based on `extract_opts` and `filter_opts`.
#     Return tagged frame with columns `token` and `pos`.
#     Columns `token` is lemmatized word or source word depending on `extract_opts.lemmatize`.

#     Args:
#         extract_opts (ExtractTaggedTokensOpts): Part-of-speech/lemma extract options (e.g. PoS-filter)
#         token2id (Token2Id, optional): Vocabulary.
#         filter_opts (PropertyValueMaskingOpts, optional): Filter based on boolean flags in tagged frame. Defaults to None.
#         text_column (str, optional): Name of text column in data frame. Defaults to 'text'.
#         lemma_column (str, optional): Name of `lemma` column in data frame. Defaults to 'lemma_'.
#         pos_column (str, optional): Name of PoS column. Defaults to 'pos_'.

#     Returns:
#         Iterable[str]: Sequence of extracted tokens
#     """
#     to_lower: bool = transform_opts and transform_opts.to_lower

#     if extract_opts.lemmatize is None and extract_opts.target_override is None:
#         raise ValueError("a valid target not supplied (no lemmatize or target")

#     if pos_column not in tagged_frame.columns:
#         raise ValueError(f"configuration error: {pos_column} not in document")

#     target: str = extract_opts.get_target_column(lemma_column, text_column)

#     if target not in tagged_frame.columns:
#         raise ValueError(f"{target} is not valid target for given document (missing column)")

#     passthroughs: Set[str] = extract_opts.get_passthrough_tokens()
#     blocks: Set[str] = extract_opts.get_block_tokens().union('')

#     if extract_opts.lemmatize or to_lower:
#         tagged_frame[target] = pd.Series([x.lower() for x in tagged_frame[target]])
#         passthroughs = {x.lower() for x in passthroughs}

#     # if extract_opts.block_chars:
#     #     for char in extract_opts.block_chars:
#     #         doc[target] = doc[target].str.replace(char, '', regex=False)

#     """ Phrase detection """
#     if extract_opts.phrases is not None:
#         found_phrases = detect_phrases(tagged_frame[target], extract_opts.phrases, ignore_case=to_lower)
#         if found_phrases:
#             tagged_frame = merge_phrases(tagged_frame, found_phrases, target_column=target, pad=PHRASE_PAD)
#             passthroughs = passthroughs.union({'_'.join(x[1]) for x in found_phrases})

#     mask = np.repeat(True, len(tagged_frame.index))
#     if filter_opts is not None:
#         mask &= filter_opts.mask(tagged_frame)

#     if len(extract_opts.get_pos_includes()) > 0:
#         """Don't filter if PoS-include is empty - and don't filter out PoS tokens that should be padded"""
#         mask &= tagged_frame[pos_column].isin(extract_opts.get_pos_includes().union(extract_opts.get_pos_paddings()))

#     if len(extract_opts.get_pos_excludes()) > 0:
#         mask &= ~(tagged_frame[pos_column].isin(extract_opts.get_pos_excludes()))

#     if transform_opts:
#         mask &= transform_opts.mask(tagged_frame[target])

#     if len(passthroughs) > 0:
#         # TODO: #52 Make passthrough token case-insensative
#         mask |= tagged_frame[target].isin(passthroughs)

#     if len(blocks) > 0:
#         mask &= ~tagged_frame[target].isin(blocks)

#     filtered_data: pd.DataFrame = tagged_frame.loc[mask][[target, pos_column]]  # .copy()

#     if extract_opts.global_tf_threshold > 1:
#         filtered_data = filter_tagged_frame_by_term_frequency(
#             tagged_frame=filtered_data,
#             target=target,
#             token2id=token2id,
#             extract_opts=extract_opts,
#             passthroughs=passthroughs,
#         )

#     filtered_data.rename(columns={target: 'token'}, inplace=True)

#     return filtered_data


def filter_tagged_frame_by_term_frequency(  # pylint: disable=too-many-arguments, too-many-statements
    tagged_frame: pd.DataFrame,
    target: str,
    token2id: Token2Id,
    extract_opts: ExtractTaggedTokensOpts,
    passthroughs: Set[str] = None,
) -> Iterable[str]:
    """Filter tagged frame `doc` based on `extract_opts` and `filter_opts`.
    Return tagged frame with columns `token` and `pos`.
    Columns `token` is lemmatized word or source word depending on `extract_opts.lemmatize`.

    Args:
        tagged_frame (pd.DataFrame): tagged frame to be filtered
        extract_opts (ExtractTaggedTokensOpts): Part-of-speech/lemma extract options (e.g. PoS-filter)
        token2id (Token2Id, optional): Vocabulary.
        filter_opts (PropertyValueMaskingOpts, optional): Filter based on boolean flags in tagged frame. Defaults to None.
        text_column (str, optional): Name of text column in data frame. Defaults to 'text'.
        lemma_column (str, optional): Name of `lemma` column in data frame. Defaults to 'lemma_'.
        pos_column (str, optional): Name of PoS column. Defaults to 'pos_'.

    Returns:
        Iterable[str]: Sequence of extracted tokens
    """

    if extract_opts.global_tf_threshold <= 1:
        return tagged_frame

    if token2id is None or token2id.tf is None:
        raise ValueError("token2id or token2id.tf is not defined")

    if target not in tagged_frame.columns:
        raise ValueError(f"{target} is not valid target for given document (missing column)")

    """
    If global_tf_threshold_mask then filter out tokens below threshold
    Otherwise replace token with `GLOBAL_TF_THRESHOLD_MASK_TOKEN`

    Alternativ implementation:
        1. Compress Token2Id (removed low frequency words)
        2. Remove or mask tokens not in compressed token2id
    """

    tg = token2id.get
    cg = token2id.tf.get

    tagged_frame['token_id'] = tagged_frame[target].apply(tg)
    tagged_frame['token_count'] = tagged_frame.token_id.apply(cg)

    low_frequency_mask = tagged_frame.token_count.fillna(0) < extract_opts.global_tf_threshold

    passthrough_ids: Set[int] = set() if not passthroughs else {tg(w) for w in passthroughs}
    if passthrough_ids:
        low_frequency_mask &= ~tagged_frame.token_id.isin(passthrough_ids)

    if extract_opts.global_tf_threshold_mask:
        """Mask low frequency terms"""
        tagged_frame[target] = tagged_frame[target].where(~low_frequency_mask, GLOBAL_TF_THRESHOLD_MASK_TOKEN)
    else:
        """Filter out low frequency terms"""
        tagged_frame = tagged_frame[~low_frequency_mask]

    return tagged_frame


def tagged_frame_to_tokens(  # pylint: disable=too-many-arguments, too-many-statements
    doc: pd.DataFrame,
    extract_opts: ExtractTaggedTokensOpts,
    token2id: Token2Id = None,
    filter_opts: PropertyValueMaskingOpts = None,
    text_column: str = 'text',
    lemma_column: str = 'lemma_',
    pos_column: str = 'pos_',
    transform_opts: TokensTransformOpts = None,
    pos_schema: PoS_Tag_Scheme = None,
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
    pos_paddings: Set[str] = extract_opts.get_pos_paddings()
    phrase_pad: str = PHRASE_PAD
    passthroughs: Set[str] = extract_opts.get_passthrough_tokens()

    filtered_data = filter_tagged_frame(
        tagged_frame=doc,
        extract_opts=extract_opts,
        token2id=token2id,
        filter_opts=filter_opts,
        text_column=text_column,
        lemma_column=lemma_column,
        pos_column=pos_column,
        transform_opts=transform_opts,
        pos_schema=pos_schema,
    )

    token_pos_tuples = filtered_data[['token', pos_column]].itertuples(index=False, name=None)

    if len(pos_paddings) > 0:
        token_pos_tuples = (
            (pad, x[1]) if x[1] in pos_paddings and x[0] not in passthroughs else x for x in token_pos_tuples
        )

    if extract_opts.append_pos:
        return [
            pad if x[0] == pad else f"{x[0].replace(' ', '_')}@{x[1]}" for x in token_pos_tuples if x[0] != phrase_pad
        ]

    if phrase_pad:
        return [x[0].replace(' ', '_') for x in token_pos_tuples if x[0] != phrase_pad]

    return [x[0] for x in token_pos_tuples]


# def filter_id_tagged_frame(  # pylint: disable=too-many-arguments, too-many-statements
#     id_tagged_frame: pd.DataFrame,
#     extract_opts: ExtractTaggedTokensOpts,
#     token2id: Token2Id,
#     pos_schema: PoS_Tag_Scheme,
#     transform_opts: TokensTransformOpts = None,
# ) -> Iterable[str]:
#     """Filter `tagged_frame` based on `extract_opts` and `transform_opts`.
#     """

#     if extract_opts.lemmatize is None and extract_opts.target_override is None:
#         raise ValueError("a valid target not supplied (no lemmatize or target")

#     if 'pos_id' not in id_tagged_frame.columns:
#         raise ValueError("pos_id not in document")

#     if 'token' not in id_tagged_frame.columns:
#         raise ValueError("token is not valid target for given document (missing column)")

#     ig = token2id.id2token.get
#     pg = pos_schema.pos_to_id.get

#     passthrough_ids: Set[str] = {ig(x) for x in extract_opts.get_passthrough_tokens()}
#     block_ids: Set[str] = {ig(x) for x in extract_opts.get_block_tokens().union('')}

#     pos_include_ids = [pg(x) for x in extract_opts.get_pos_includes()]
#     pos_exclude_ids = [pg(x) for x in extract_opts.get_pos_excludes()]
#     pos_padding_ids = [pg(x) for x in extract_opts.get_pos_paddings()]

#     """ Phrase detection """
#     # if extract_opts.phrases is not None:
#     #     found_phrases = detect_phrases(tagged_frame[target], extract_opts.phrases, ignore_case=to_lower)
#     #     if found_phrases:
#     #         tagged_frame = merge_phrases(tagged_frame, found_phrases, target_column=target, pad=PHRASE_PAD)

#     mask = np.repeat(True, len(id_tagged_frame.index))

#     if len(pos_include_ids) > 0:
#         """Don't filter if PoS-include is empty - and don't filter out PoS tokens that should be padded"""
#         mask &= id_tagged_frame.pos_id.isin(pos_include_ids.union(pos_padding_ids))

#     if len(pos_exclude_ids) > 0:
#         mask &= ~(id_tagged_frame.pos_id.isin(pos_exclude_ids))

#     if transform_opts:
#         mask &= transform_opts.mask(id_tagged_frame.token_id)

#     if len(passthrough_ids) > 0:
#         mask |= id_tagged_frame.token_id.isin(passthrough_ids)

#     if len(block_ids) > 0:
#         mask &= ~id_tagged_frame.token_id.isin(block_ids)

#     filtered_data: pd.DataFrame = id_tagged_frame.loc[mask][['token_id', 'pos_id']]  # .copy()

#     # if extract_opts.global_tf_threshold > 1:
#     #     filtered_data = filter_tagged_frame_by_term_frequency(
#     #         tagged_frame=filtered_data,
#     #         target=target,
#     #         token2id=token2id,
#     #         extract_opts=extract_opts,
#     #         passthroughs=passthroughs,
#     #     )

#     return filtered_data


def filter_tagged_frame(  # pylint: disable=too-many-arguments, too-many-statements
    tagged_frame: pd.DataFrame,
    *,
    extract_opts: ExtractTaggedTokensOpts,
    token2id: Token2Id = None,
    pos_schema: PoS_Tag_Scheme = None,
    filter_opts: PropertyValueMaskingOpts = None,
    text_column: str = 'text',
    lemma_column: str = 'lemma_',
    pos_column: str = 'pos_',
    transform_opts: TokensTransformOpts = None,
) -> Iterable[str]:
    """Filter tagged frame `doc` based on `extract_opts` and `filter_opts`.
    Return tagged frame with columns `token` and `pos`.
    Columns `token` is lemmatized word or source word depending on `extract_opts.lemmatize`.

    Args:
        extract_opts (ExtractTaggedTokensOpts): Part-of-speech/lemma extract options (e.g. PoS-filter)
        token2id (Token2Id, optional): Vocabulary.
        filter_opts (PropertyValueMaskingOpts, optional): Filter based on boolean flags in tagged frame. Defaults to None.
        text_column (str, optional): Name of text column in data frame. Defaults to 'text'.
        lemma_column (str, optional): Name of `lemma` column in data frame. Defaults to 'lemma_'.
        pos_column (str, optional): Name of PoS column. Defaults to 'pos_'.

    Returns:
        Iterable[str]: Sequence of extracted tokens
    """

    is_numeric_frame: bool = 'pos_id' in tagged_frame.columns and 'token_id' in tagged_frame.columns

    to_lower: bool = transform_opts and transform_opts.to_lower

    if is_numeric_frame:
        if to_lower:
            logger.warning("lowercasing not implemented for numeric tagged frames")
            to_lower = False

    if not is_numeric_frame and extract_opts.lemmatize is None and extract_opts.target_override is None:
        raise ValueError("a valid target not supplied (no lemmatize or target")

    if is_numeric_frame:
        target = 'token_id'
        pos_column = 'pos_id'
    else:
        target: str = extract_opts.get_target_column(lemma_column, text_column)

    if target not in tagged_frame.columns:
        raise ValueError(f"{target} is not valid target for given document (missing column)")

    if pos_column not in tagged_frame.columns:
        raise ValueError(f"configuration error: {pos_column} not in document")

    passthroughs: Set[str] = extract_opts.get_passthrough_tokens()
    blocks: Set[str] = extract_opts.get_block_tokens().union('')

    if is_numeric_frame:
        passthroughs = token2id.to_id_set(passthroughs)
        blocks = token2id.to_id_set(blocks)

    if not is_numeric_frame and extract_opts.lemmatize or to_lower:
        tagged_frame[target] = pd.Series([x.lower() for x in tagged_frame[target]])
        passthroughs = {x.lower() for x in passthroughs}

    # if extract_opts.block_chars:
    #     for char in extract_opts.block_chars:
    #         doc[target] = doc[target].str.replace(char, '', regex=False)

    """ Phrase detection """
    if not is_numeric_frame and extract_opts.phrases is not None:
        if is_numeric_frame:
            logger.warning("phrase detection not implemented for numeric tagged frames")
        else:
            found_phrases = detect_phrases(tagged_frame[target], extract_opts.phrases, ignore_case=to_lower)
            if found_phrases:
                tagged_frame = merge_phrases(tagged_frame, found_phrases, target_column=target, pad=PHRASE_PAD)
                passthroughs = passthroughs.union({'_'.join(x[1]) for x in found_phrases})

    mask = np.repeat(True, len(tagged_frame.index))
    if filter_opts is not None:
        mask &= filter_opts.mask(tagged_frame)

    pos_includes = extract_opts.get_pos_includes()
    pos_excludes = extract_opts.get_pos_excludes()
    pos_paddings = extract_opts.get_pos_paddings()

    if is_numeric_frame:
        pg = pos_schema.pos_to_id.get
        pos_includes = {pg(x) for x in pos_includes}
        pos_excludes = {pg(x) for x in pos_excludes}
        pos_paddings = {pg(x) for x in pos_paddings}

    if len(pos_includes) > 0:
        """Don't filter if PoS-include is empty - and don't filter out PoS tokens that should be padded"""
        mask &= tagged_frame[pos_column].isin(pos_includes.union(pos_paddings))

    if len(pos_excludes) > 0:
        mask &= ~(tagged_frame[pos_column].isin(pos_excludes))

    if transform_opts:
        mask &= transform_opts.mask(tagged_frame[target])

    if len(passthroughs) > 0:
        mask |= tagged_frame[target].isin(passthroughs)

    if len(blocks) > 0:
        mask &= ~tagged_frame[target].isin(blocks)

    filtered_data: pd.DataFrame = tagged_frame.loc[mask][[target, pos_column]]

    if extract_opts.global_tf_threshold > 1:
        if is_numeric_frame:
            logger.warning("TF filter not implemented for numeric tagged frames")
        else:
            filtered_data = filter_tagged_frame_by_term_frequency(
                tagged_frame=filtered_data,
                target=target,
                token2id=token2id,
                extract_opts=extract_opts,
                passthroughs=passthroughs,
            )

    if not is_numeric_frame:

        filtered_data.rename(columns={target: 'token'}, inplace=True)

    return filtered_data
