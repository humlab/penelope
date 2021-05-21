import os
from typing import Optional

import penelope.co_occurrence as co_occurrence
import penelope.pipeline as pipeline
from loguru import logger

from penelope.co_occurrence import co_occurrences_to_co_occurrence_corpus
from penelope.corpus import VectorizedCorpus
from penelope.notebook import interface

POS_CHECKPOINT_FILENAME_POSTFIX = '_pos_tagged_frame_csv.zip'


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

        checkpoint_filename: Optional[str] = (
            checkpoint_file or f"{corpus_config.corpus_name}_{POS_CHECKPOINT_FILENAME_POSTFIX}"
        )

        tagged_frame_pipeline: pipeline.CorpusPipeline = corpus_config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_filename=args.corpus_filename,
            checkpoint_filename=checkpoint_filename,
        )

        args.extract_opts.passthrough_tokens = list(args.context_opts.concept)

        p: pipeline.CorpusPipeline = (
            tagged_frame_pipeline
            + pipeline.wildcard_to_partition_by_document_co_occurrence_pipeline(
                transform_opts=args.transform_opts,
                extract_opts=args.extract_opts,
                filter_opts=args.filter_opts,
                context_opts=args.context_opts,
                global_threshold_count=args.count_threshold,
            )
        )

        value: co_occurrence.CoOccurrenceComputeResult = p.value()

        if len(value.co_occurrences) == 0:
            raise co_occurrence.ZeroComputeError()

        corpus: VectorizedCorpus = co_occurrences_to_co_occurrence_corpus(
            co_occurrences=value.co_occurrences,
            document_index=value.document_index,
            # legacy: partition_key=args.context_opts.partition_keys[0],
            token2id=value.token2id,
        )

        bundle = co_occurrence.Bundle(
            corpus=corpus,
            tag=args.corpus_tag,
            folder=args.target_folder,
            co_occurrences=value.co_occurrences,
            document_index=value.document_index,
            token2id=value.token2id,
            token_window_counts=value.token_window_counts,
            compute_options=co_occurrence.create_options_bundle(
                reader_opts=corpus_config.text_reader_opts,
                transform_opts=args.transform_opts,
                context_opts=args.context_opts,
                extract_opts=args.extract_opts,
                input_filename=args.corpus_filename,
                output_filename=target_filename,
                count_threshold=args.count_threshold,
            ),
        )

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
