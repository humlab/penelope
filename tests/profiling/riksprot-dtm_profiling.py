from __future__ import annotations

from typing import Optional, Sequence

import penelope.workflows.vectorize.dtm_id as workflow
from penelope import corpus as pc
from penelope import pipeline as pp
from penelope import utility
from penelope.pipeline.phrases import parse_phrases
from penelope.utility.pos_tags import pos_tags_to_str
from penelope.workflows.vectorize.dtm import store_corpus_bundle

# import cProfile
# import pstats
# pylint: disable=too-many-arguments

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
    workflow.compute(args=args, corpus_config=corpus_config)
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
        pp.CorpusPipeline(config=corpus_config)
        .load_id_tagged_frame(
            folder=corpus_source,
            id_to_token=id_to_token,
            file_pattern=file_pattern,
        )
        .filter_tagged_frame(
            extract_opts=extract_opts,
            pos_schema=corpus_config.pos_schema,
            transform_opts=transform_opts,
        )
        .take(n_count=10000)
        .to_dtm(vectorize_opts=vectorize_opts, tagged_column=extract_opts.target_column)
    )

    # for payload in pipeline.resolve():
    #     pass

    # pipeline.exhaust(2000)

    corpus: pc.VectorizedCorpus = pipeline.value()

    if (args.tf_threshold or 1) > 1:
        corpus = corpus.slice_by_tf(args.tf_threshold)

    if args.persist:
        store_corpus_bundle(corpus, args)

    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats.print_stats()


if __name__ == '__main__':
    main()
