from __future__ import annotations

from typing import Iterable, Set

import numpy as np
import pandas as pd
from loguru import logger

from penelope.corpus import Token2Id, TokensTransformOpts
from penelope.corpus.readers import GLOBAL_TF_THRESHOLD_MASK_TOKEN, ExtractTaggedTokensOpts
from penelope.utility import PoS_Tag_Scheme

from .phrases import PHRASE_PAD, detect_phrases, merge_phrases


def is_encoded_tagged_frame(tagged_frame: pd.DataFrame) -> bool:
    is_numeric_frame: bool = 'pos_id' in tagged_frame.columns and (
        'token_id' in tagged_frame.columns or 'lemma_id' in tagged_frame.columns
    )
    return is_numeric_frame


class Token2IdMissingError(ValueError):
    ...


class PoSTagSchemaMissingError(ValueError):
    ...


class TaggedFrameColumnNameError(ValueError):
    ...


# pylint: disable=too-many-arguments, too-many-statements


def filter_tagged_frame(
    tagged_frame: pd.DataFrame,
    *,
    extract_opts: ExtractTaggedTokensOpts,
    token2id: Token2Id = None,
    pos_schema: PoS_Tag_Scheme = None,
    normalize_column_names: bool = True,
    transform_opts: TokensTransformOpts = None,
) -> pd.DataFrame:
    """Filters tagged frame (text or numeric). Returns tagged frame

    Args:
        tagged_frame ([pd.DataFrame]): Document frame to be filtered, can be text or numeric
        extract_opts (ExtractTaggedTokensOpts): PoS and lemma extract/filter opts
        token2id (Token2Id, optional): Vocabulary. Defaults to None.
        pos_schema (PoS_Tag_Scheme, optional): PoS schema. Defaults to None.
        transform_opts (TokensTransformOpts, optional): Filters and transforms. Defaults to None.
        normalize_column_names (bool, optional): If text, rename columns to `token` and `pos`. Defaults to True.

    Raises:
        Token2IdMissingError: Token2Id is mandatory if frame is numeric.
        PoSTagSchemaMissingError: PoS-schema is mandatory if frame is numeric.
        TaggedFrameColumnNameError: Missing target column (corrupt data)

    Returns:
        pd.DataFrame: Filtered and transformed document frame.
    """
    if len(tagged_frame) == 0:
        return []

    is_numeric_frame: bool = is_encoded_tagged_frame(tagged_frame)
    to_lower: bool = transform_opts and transform_opts.to_lower

    if is_numeric_frame:

        if token2id is None:
            raise Token2IdMissingError("filter_tagged_frame: cannot filter tagged id frame without vocabulary")

        if pos_schema is None:
            raise PoSTagSchemaMissingError("filter_tagged_frame: cannot filter tagged id frame without pos_schema")

        if to_lower:
            logger.warning("lowercasing not implemented for numeric tagged frames")
            to_lower = False

    if not is_numeric_frame and extract_opts.lemmatize is None and extract_opts.target_override is None:
        raise ValueError("a valid target not supplied (no lemmatize or target")

    target_column: str = extract_opts.target_column
    pos_column: str = extract_opts.pos_column

    if target_column not in tagged_frame.columns:
        raise TaggedFrameColumnNameError(f"{target_column} is not valid target for given document (missing column)")

    if pos_column not in tagged_frame.columns:
        raise ValueError(f"configuration error: {pos_column} not in document")

    passthroughs: Set[str] = extract_opts.get_passthrough_tokens()
    blocks: Set[str] = extract_opts.get_block_tokens().union('')

    if is_numeric_frame:
        passthroughs = token2id.to_id_set(passthroughs)
        blocks = token2id.to_id_set(blocks)

    if not is_numeric_frame and (extract_opts.lemmatize or to_lower):
        tagged_frame[target_column] = tagged_frame[target_column].str.lower()
        # pd.Series([x.lower() for x in tagged_frame[target_column]])
        passthroughs = {x.lower() for x in passthroughs}

    # if extract_opts.block_chars:
    #     for char in extract_opts.block_chars:
    #         doc[target] = doc[target].str.replace(char, '', regex=False)

    """ Phrase detection """
    if extract_opts.phrases:
        if is_numeric_frame:
            logger.warning("phrase detection not implemented for numeric tagged frames")
            extract_opts.phrases = None
        else:
            found_phrases = detect_phrases(tagged_frame[target_column], extract_opts.phrases, ignore_case=to_lower)
            if found_phrases:
                tagged_frame = merge_phrases(tagged_frame, found_phrases, target_column=target_column, pad=PHRASE_PAD)
                passthroughs = passthroughs.union({'_'.join(x[1]) for x in found_phrases})

    mask = np.repeat(True, len(tagged_frame.index))
    if extract_opts.filter_opts and extract_opts.filter_opts.data:
        mask &= extract_opts.filter_opts.mask(tagged_frame)

    pos_includes: Set[str] = extract_opts.get_pos_includes()
    pos_excludes: Set[str] = extract_opts.get_pos_excludes()
    pos_paddings: Set[str] = extract_opts.get_pos_paddings()

    if is_numeric_frame:
        pg = pos_schema.pos_to_id.get
        pos_includes = {pg(x) for x in pos_includes}
        pos_excludes = {pg(x) for x in pos_excludes}
        pos_paddings = {pg(x) for x in pos_paddings}

    if pos_includes:
        """Don't filter if PoS-include is empty - and don't filter out PoS tokens that should be padded"""
        mask &= tagged_frame[pos_column].isin(pos_includes.union(pos_paddings))

    if pos_excludes:
        mask &= ~(tagged_frame[pos_column].isin(pos_excludes))

    if transform_opts and transform_opts.has_effect:
        mask &= transform_opts.mask(tagged_frame[target_column], token2id=token2id)

    if len(passthroughs) > 0:
        mask |= tagged_frame[target_column].isin(passthroughs)

    if len(blocks) > 0:
        mask &= ~tagged_frame[target_column].isin(blocks)

    filtered_data: pd.DataFrame = tagged_frame.loc[mask][[target_column, pos_column]]

    if extract_opts.global_tf_threshold > 1:
        if token2id is None or token2id.tf is None:
            logger.error("Cannot apply TF filter since token2id has no term frequencies")
            extract_opts.global_tf_threshold = 1
        else:
            filtered_data = filter_tagged_frame_by_term_frequency(
                tagged_frame=filtered_data,
                target_column=target_column,
                token2id=token2id,
                extract_opts=extract_opts,
                passthroughs=passthroughs,
            )

    if not is_numeric_frame and normalize_column_names:

        filtered_data.rename(columns={target_column: 'token', pos_column: 'pos'}, inplace=True)

    return filtered_data


def filter_tagged_frame_by_term_frequency(  # pylint: disable=too-many-arguments, too-many-statements
    tagged_frame: pd.DataFrame,
    target_column: str,
    token2id: Token2Id,
    extract_opts: ExtractTaggedTokensOpts,
    passthroughs: Set[str] = None,
) -> pd.DataFrame:
    """Filter tagged frame `tagged_frame` based on `extract_opts`.
    Return tagged frame with columns `token` and `pos`.
    Columns `token` is lemmatized word or source word depending on `extract_opts.lemmatize`.

    Args:
        tagged_frame (pd.DataFrame): tagged frame to be filtered
        extract_opts (ExtractTaggedTokensOpts): Part-of-speech/lemma extract options (e.g. PoS-filter)
        token2id (Token2Id, optional): Vocabulary.

    Returns:
        pd.DataFrame: Filtered tagged frame
    """

    is_numeric_frame: bool = is_encoded_tagged_frame(tagged_frame)

    if extract_opts.global_tf_threshold <= 1:
        return tagged_frame

    if token2id is None or token2id.tf is None:
        raise ValueError("token2id or token2id.tf is not defined")

    if target_column not in tagged_frame.columns:
        raise ValueError(f"{target_column} is not valid target for given document (missing column)")

    """
    If global_tf_threshold_mask then filter out tokens below threshold
    Otherwise replace token with `GLOBAL_TF_THRESHOLD_MASK_TOKEN`

    Alternativ implementation:
        1. Compress Token2Id (remove low frequency words)
        2. Remove or mask tokens not in compressed token2id
    """

    tg = token2id.get

    mask_token_id: int = tg(GLOBAL_TF_THRESHOLD_MASK_TOKEN)
    mask_token: str | int = mask_token_id if is_numeric_frame else GLOBAL_TF_THRESHOLD_MASK_TOKEN

    """Set low TF to 0"""
    mask_token_id_tf: int = token2id.tf.get(mask_token_id, 0)
    token2id.tf[mask_token_id] = 0

    cg = token2id.tf.get

    low_frequency_mask: pd.Series = (
        tagged_frame[target_column].apply(tg).apply(cg)
        if not is_numeric_frame
        else tagged_frame[target_column].apply(cg)
    ).fillna(0) < extract_opts.global_tf_threshold

    """Reset low TF count"""
    token2id.tf[mask_token_id] = mask_token_id_tf

    if passthroughs:
        low_frequency_mask &= ~tagged_frame[target_column].isin(passthroughs)

    if extract_opts.global_tf_threshold_mask:
        """Mask low frequency terms"""
        tagged_frame[target_column] = tagged_frame[target_column].where(~low_frequency_mask, mask_token)
    else:
        """Filter out low frequency terms"""
        tagged_frame = tagged_frame[~low_frequency_mask]

    return tagged_frame


def tagged_frame_to_tokens(  # pylint: disable=too-many-arguments, too-many-statements
    doc: pd.DataFrame,
    extract_opts: ExtractTaggedTokensOpts,
    token2id: Token2Id = None,
    transform_opts: TokensTransformOpts = None,
    pos_schema: PoS_Tag_Scheme = None,
) -> Iterable[str | int]:
    """Extracts tokens from a tagged document represented as a Pandas data frame.

    Args:
        extract_opts (ExtractTaggedTokensOpts): Part-of-speech/lemma extract options (e.g. PoS-filter)

    Returns:
        Iterable[str]: Sequence of extracted tokens
    """

    if len(doc) == 0:
        return []

    is_numeric_frame: bool = is_encoded_tagged_frame(doc)

    if isinstance(extract_opts, str):
        passthrough_column: str = extract_opts
        """If extracts_opts is a string: return column with what name"""
        if transform_opts is not None:
            raise ValueError("transform_opts must be None when passthrough is specified")
        if transform_opts is not None:
            raise ValueError("transform_opts must be None when passthrough is specified")
        return doc[passthrough_column].tolist()

    # if is_numeric_frame:
    #     raise NotImplementedError("tagged_frame_to_tokens cannot handle encoded frames yet")

    pad: str = "*"
    pos_paddings: Set[str] = extract_opts.get_pos_paddings()
    phrase_pad: str = PHRASE_PAD

    filtered_data = filter_tagged_frame(
        tagged_frame=doc,
        extract_opts=extract_opts,
        token2id=token2id,
        transform_opts=transform_opts,
        pos_schema=pos_schema,
    )

    target_columns = ['token_id', 'pos_id'] if is_numeric_frame else ['token', 'pos']
    token_pos_tuples = filtered_data[target_columns].itertuples(index=False, name=None)

    if len(pos_paddings) > 0:

        passthroughs: Set[str] = extract_opts.get_passthrough_tokens()

        if is_numeric_frame:
            pos_paddings = {pos_schema.pos_to_id.get(x) for x in pos_paddings}
            passthroughs = token2id.to_id_set(passthroughs)

        token_pos_tuples = (
            (pad, x[1]) if x[1] in pos_paddings and x[0] not in passthroughs else x for x in token_pos_tuples
        )

    if extract_opts.append_pos:

        if is_numeric_frame:
            raise NotImplementedError("tagged_frame_to_tokens: PoS tag on encoded frame")

        return [
            pad if x[0] == pad else f"{x[0].replace(' ', '_')}@{x[1]}" for x in token_pos_tuples if x[0] != phrase_pad
        ]

    if not is_numeric_frame and extract_opts.phrases and len(extract_opts.phrases) > 0:
        return [x[0].replace(' ', '_') for x in token_pos_tuples if x[0] != phrase_pad]

    return [x[0] for x in token_pos_tuples]
