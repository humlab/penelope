from __future__ import annotations

from typing import Optional, Sequence, Set

import numpy as np
import pandas as pd
from loguru import logger

import penelope.workflows.vectorize.dtm_id as workflow
from penelope import corpus as pc
from penelope import pipeline as pp
from penelope import utility
from penelope.corpus import Token2Id, TokensTransformOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.pipeline import convert
from penelope.pipeline.phrases import PHRASE_PAD, detect_phrases, merge_phrases, parse_phrases
from penelope.utility import PoS_Tag_Scheme
from penelope.utility.pos_tags import pos_tags_to_str

# import cProfile
# import pstats


OPTIONS_FILENAME = "tests/profiling/riksprot-1965_dtm_opts.yml"


def create_compute_opts(
    corpus_config: pp.CorpusConfig,
    corpus_source: Optional[str] = None,
    output_folder: Optional[str] = None,
    output_tag: Optional[str] = None,
    filename_pattern: Optional[str] = None,
    phrase: Sequence[str] = None,
    phrase_file: Optional[str] = None,
    create_subfolder: bool = True,
    pos_includes: Optional[str] = None,
    pos_paddings: Optional[str] = None,
    pos_excludes: Optional[str] = None,
    append_pos: bool = False,
    max_tokens: int = None,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: Optional[str] = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    deserialize_processes: int = 4,
):

    phrases: dict = parse_phrases(phrase_file, phrase)

    if pos_excludes is None:
        pos_excludes = pos_tags_to_str(corpus_config.pos_schema.Delimiter)

    if pos_paddings and pos_paddings.upper() in ["FULL", "ALL", "PASSTHROUGH"]:
        pos_paddings = pos_tags_to_str(corpus_config.pos_schema.all_types_except(pos_includes))

    text_reader_opts: pc.TextReaderOpts = corpus_config.text_reader_opts.copy()

    if filename_pattern is not None:
        text_reader_opts.filename_pattern = filename_pattern

    corpus_config.checkpoint_opts.deserialize_processes = max(1, deserialize_processes)

    tagged_columns: dict = corpus_config.pipeline_payload.tagged_columns_names
    args: workflow.ComputeOpts = workflow.ComputeOpts(
        corpus_type=corpus_config.corpus_type,
        corpus_source=corpus_source,
        target_folder=output_folder,
        corpus_tag=output_tag,
        tf_threshold=tf_threshold,
        tf_threshold_mask=tf_threshold_mask,
        create_subfolder=create_subfolder,
        persist=True,
        filename_pattern=filename_pattern,
        transform_opts=pc.TokensTransformOpts(
            to_lower=to_lower,
            to_upper=False,
            min_len=min_word_length,
            max_len=max_word_length,
            remove_accents=False,
            remove_stopwords=(remove_stopwords is not None),
            stopwords=None,
            extra_stopwords=None,
            language=remove_stopwords,
            keep_numerals=keep_numerals,
            keep_symbols=keep_symbols,
        ),
        extract_opts=pc.ExtractTaggedTokensOpts(
            pos_includes=pos_includes,
            pos_paddings=pos_paddings,
            pos_excludes=pos_excludes,
            lemmatize=lemmatize,
            phrases=phrases,
            append_pos=append_pos,
            global_tf_threshold=tf_threshold,
            global_tf_threshold_mask=tf_threshold_mask,
            **tagged_columns,
        ),
        vectorize_opts=pc.VectorizeOpts(
            already_tokenized=True,
            min_tf=tf_threshold,
            max_tokens=max_tokens,
        ),
    )
    return args


def is_encoded_tagged_frame(tagged_frame: pd.DataFrame) -> bool:
    is_numeric_frame: bool = 'pos_id' in tagged_frame.columns and (
        'token_id' in tagged_frame.columns or 'lemma_id' in tagged_frame.columns
    )
    return is_numeric_frame


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
    # if len(tagged_frame) == 0:
    #     return []

    is_numeric_frame: bool = is_encoded_tagged_frame(tagged_frame)
    # to_lower: bool = transform_opts and transform_opts.to_lower

    # if is_numeric_frame:

    #     if token2id is None:
    #         raise ValueError("filter_tagged_frame: cannot filter tagged id frame without vocabulary")

    #     if pos_schema is None:
    #         raise ValueError("filter_tagged_frame: cannot filter tagged id frame without pos_schema")

    #     if to_lower:
    #         logger.warning("lowercasing not implemented for numeric tagged frames")
    #         to_lower = False

    # if not is_numeric_frame and extract_opts.lemmatize is None and extract_opts.target_override is None:
    #     raise ValueError("a valid target not supplied (no lemmatize or target")

    target_column: str = extract_opts.target_column
    pos_column: str = extract_opts.pos_column

    # if target_column not in tagged_frame.columns:
    #     raise ValueError(f"{target_column} is not valid target for given document (missing column)")

    # if pos_column not in tagged_frame.columns:
    #     raise ValueError(f"configuration error: {pos_column} not in document")

    passthroughs: Set[str] = extract_opts.get_passthrough_tokens()
    blocks: Set[str] = extract_opts.get_block_tokens().union('')

    if is_numeric_frame:
        passthroughs = token2id.to_id_set(passthroughs)
        blocks = token2id.to_id_set(blocks)

    # if not is_numeric_frame and (extract_opts.lemmatize or to_lower):
    #     tagged_frame[target_column] = tagged_frame[target_column].str.lower()
    #     # pd.Series([x.lower() for x in tagged_frame[target_column]])
    #     passthroughs = {x.lower() for x in passthroughs}

    # # if extract_opts.block_chars:
    # #     for char in extract_opts.block_chars:
    # #         doc[target] = doc[target].str.replace(char, '', regex=False)

    """ Phrase detection """
    # if extract_opts.phrases:
    #     if is_numeric_frame:
    #         logger.warning("phrase detection not implemented for numeric tagged frames")
    #         extract_opts.phrases = None
    #     else:
    #         found_phrases = detect_phrases(tagged_frame[target_column], extract_opts.phrases, ignore_case=to_lower)
    #         if found_phrases:
    #             tagged_frame = merge_phrases(tagged_frame, found_phrases, target_column=target_column, pad=PHRASE_PAD)
    #             passthroughs = passthroughs.union({'_'.join(x[1]) for x in found_phrases})

    mask = np.repeat(True, len(tagged_frame.index))
    # if extract_opts.filter_opts and extract_opts.filter_opts.data:
    #     mask &= extract_opts.filter_opts.mask(tagged_frame)

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

    # if extract_opts.global_tf_threshold > 1:
    #     if token2id is None or token2id.tf is None:
    #         logger.error("Cannot apply TF filter since token2id has no term frequencies")
    #         extract_opts.global_tf_threshold = 1
    #     else:
    #         filtered_data = convert.filter_tagged_frame_by_term_frequency(
    #             tagged_frame=filtered_data,
    #             target_column=target_column,
    #             token2id=token2id,
    #             extract_opts=extract_opts,
    #             passthroughs=passthroughs,
    #         )

    # if not is_numeric_frame and normalize_column_names:

    #     filtered_data.rename(columns={target_column: 'token', pos_column: 'pos'}, inplace=True)

    return filtered_data


def main():

    # config_filename: str = 'tests/profiling/riksprot-1965_corpus_config.yml'
    config_filename: str = '/data/riksdagen_corpus_data/tagged_frames_v0.4.1_speeches.feather/corpus.yml'
    corpus_source: str = '/data/riksdagen_corpus_data/tagged_frames_v0.4.1_speeches.feather'

    corpus_config: pp.CorpusConfig = pp.CorpusConfig.load(config_filename).folders(corpus_source, method='replace')

    arguments = utility.update_dict_from_yaml(
        OPTIONS_FILENAME,
        {
            'corpus_source': corpus_source,
            'output_folder': './tests/output/',
            'output_tag': 'dtm_1965_cprofile',
            'filename_pattern': "**/prot-*.feather",
            # 'filename_pattern': "**/prot-1965*.feather",
        },
    )
    args: workflow.ComputeOpts = create_compute_opts(corpus_config=corpus_config, **arguments)

    # profiler = cProfile.Profile()
    # profiler.enable()

    assert args.is_satisfied()

    corpus_source = args.corpus_source
    extract_opts = args.extract_opts
    file_pattern = args.filename_pattern
    id_to_token = False
    transform_opts = args.transform_opts
    vectorize_opts = args.vectorize_opts

    args.extract_opts.global_tf_threshold = 1

    if corpus_source is None:
        corpus_source = corpus_config.pipeline_payload.source

    extract_opts.set_numeric_names()
    vectorize_opts.min_df = extract_opts.global_tf_threshold
    extract_opts.global_tf_threshold = 1

    pipeline: pp.CorpusPipeline = (
        pp.CorpusPipeline(config=corpus_config).load_id_tagged_frame(
            folder=corpus_source,
            id_to_token=id_to_token,
            file_pattern=file_pattern,
        )
        # .filter_tagged_frame(
        #     extract_opts=extract_opts,
        #     pos_schema=corpus_config.pos_schema,
        #     transform_opts=transform_opts,
        # )
        # .to_dtm(vectorize_opts=vectorize_opts, tagged_column=extract_opts.target_column)
    )

    for payload in pipeline.resolve():

        tagged_frame: pd.DataFrame = filter_tagged_frame(
            tagged_frame=payload.content,
            extract_opts=extract_opts,
            token2id=pipeline.payload.token2id,
            pos_schema=corpus_config.pos_schema,
            transform_opts=transform_opts,
            normalize_column_names=False,
        )

    # p.exhaust()

    # corpus: pc.VectorizedCorpus = p.value()

    # if (args.tf_threshold or 1) > 1:
    #     corpus = corpus.slice_by_tf(args.tf_threshold)

    # if args.persist:
    #     store_corpus_bundle(corpus, args)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()


if __name__ == '__main__':

    main()
