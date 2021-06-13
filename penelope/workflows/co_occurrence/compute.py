import os
from typing import Optional

import penelope.co_occurrence as co_occurrence
import penelope.pipeline as pipeline
from loguru import logger
from penelope.notebook import interface

POS_CHECKPOINT_FILENAME_POSTFIX = '_pos_tagged_frame_csv.zip'

jj = os.path.join
dirname = os.path.dirname

# pylint: disable=unused-argument
def compute(
    args: interface.ComputeOpts,
    corpus_config: pipeline.CorpusConfig,
    checkpoint_file: Optional[str] = None,
) -> co_occurrence.Bundle:
    """Creates and stores a concept co-occurrence bundle using specified options."""

    try:

        assert args.is_satisfied()

        target_filename = co_occurrence.to_filename(folder=args.target_folder, tag=args.corpus_tag)

        os.makedirs(args.target_folder, exist_ok=True)

        checkpoint_filename: Optional[str] = checkpoint_file or jj(
            dirname(args.corpus_filename), f"{args.corpus_tag}{POS_CHECKPOINT_FILENAME_POSTFIX}"
        )

        tagged_frame_pipeline: pipeline.CorpusPipeline = corpus_config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_filename=args.corpus_filename,
            checkpoint_filename=checkpoint_filename,
        )

        args.extract_opts.passthrough_tokens = args.context_opts.concept
        args.extract_opts.block_tokens = []
        # args.extract_opts.block_chars = ''
        args.extract_opts.global_tf_threshold = args.tf_threshold

        p: pipeline.CorpusPipeline = (
            tagged_frame_pipeline
            + pipeline.wildcard_to_partition_by_document_co_occurrence_pipeline(
                transform_opts=args.transform_opts,
                extract_opts=args.extract_opts,
                filter_opts=args.filter_opts,
                context_opts=args.context_opts,
                global_threshold_count=args.tf_threshold,
            )
        )

        bundle: co_occurrence.Bundle = p.value()

        if bundle.corpus is None:
            raise co_occurrence.ZeroComputeError()

        bundle.tag = args.corpus_tag
        bundle.folder = args.target_folder

        try:
            bundle.co_occurrences = bundle.corpus.to_co_occurrences(bundle.token2id)
        except ValueError as ex:
            logger.error("fatal: to_co_occurrences failed (skipping)")
            logger.exception(ex)

        bundle.compute_options = compile_compute_options(args, target_filename)

        bundle.store()

        return bundle

    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
    ) as ex:
        logger.error(ex)
        raise
    except Exception as ex:
        logger.error(ex)
        raise


def compile_compute_options(args: interface.ComputeOpts, target_filename: str = "") -> dict:
    return co_occurrence.create_options_bundle(
        reader_opts=args.text_reader_opts,
        transform_opts=args.transform_opts,
        context_opts=args.context_opts,
        extract_opts=args.extract_opts,
        input_filename=args.corpus_filename,
        output_filename=target_filename,
        tf_threshold=10,
        tf_threshold_mask=False,
    )
