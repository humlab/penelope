import os
import shutil
from typing import Optional

import penelope.co_occurrence as co_occurrence
import penelope.pipeline as pipeline
from penelope.corpus import VectorizedCorpus
from penelope.notebook import interface
from penelope.utility import getLogger
from penelope.utility.filename_utils import strip_extensions

logger = getLogger('penelope')

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

        target_filename = co_occurrence.folder_and_tag_to_filename(folder=args.target_folder, tag=args.corpus_tag)

        os.makedirs(args.target_folder, exist_ok=True)

        checkpoint_filename: Optional[str] = (
            checkpoint_file or f"{corpus_config.corpus_name}_{POS_CHECKPOINT_FILENAME_POSTFIX}"
        )

        feather_folder = _feather_folder_name(args)

        tagged_frame_pipeline: pipeline.CorpusPipeline = corpus_config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_filename=args.corpus_filename,
            checkpoint_filename=checkpoint_filename,
        ).checkpoint_feather(feather_folder)

        if args.force:
            shutil.rmtree(feather_folder, ignore_errors=True)

        compute_result: co_occurrence.ComputeResult = (
            tagged_frame_pipeline
            + pipeline.wildcard_to_co_occurrence_pipeline(
                tokens_transform_opts=args.tokens_transform_opts,
                extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
                tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
                context_opts=args.context_opts,
                global_threshold_count=args.count_threshold,
                partition_column=args.partition_keys[0],
            )
        ).value()

        if len(compute_result.co_occurrences) == 0:
            raise ValueError("Computation ended up in ZERO records. Check settinsgs!")

        corpus: VectorizedCorpus = co_occurrence.to_vectorized_corpus(
            co_occurrences=compute_result.co_occurrences,
            document_index=compute_result.document_index,
            value_column='value',
        )

        bundle = co_occurrence.Bundle(
            corpus=corpus,
            corpus_tag=args.corpus_tag,
            corpus_folder=args.target_folder,
            co_occurrences=compute_result.co_occurrences,
            document_index=compute_result.document_index,
            compute_options=co_occurrence.create_options_bundle(
                reader_opts=corpus_config.text_reader_opts,
                tokens_transform_opts=args.tokens_transform_opts,
                context_opts=args.context_opts,
                extract_tokens_opts=args.extract_tagged_tokens_opts,
                input_filename=args.corpus_filename,
                output_filename=target_filename,
                partition_keys=args.partition_keys,
                count_threshold=args.count_threshold,
            ),
        )

        co_occurrence.store_bundle(target_filename, bundle)

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


def _feather_folder_name(args):
    folder, filename = os.path.split(args.corpus_filename)
    feather_folder = os.path.join(folder, "shared", "checkpoints", f'{strip_extensions(filename)}_feather')
    return feather_folder
