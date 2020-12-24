import os
from typing import Callable, Optional

import penelope.co_occurrence as co_occurrence
from penelope.corpus import VectorizedCorpus
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility import getLogger

from ..utility import default_done_callback
from .to_co_occurrence_gui import ComputeGUI

logger = getLogger('penelope')
jj = os.path.join

POS_CHECKPOINT_FILENAME_POSTFIX = '_spaCy_pos_tagged_frame_csv.zip'


# pylint: disable=unused-argument
def compute_co_occurrence(
    corpus_config: CorpusConfig,
    args: ComputeGUI,
    partition_key: str,
    done_callback: Callable,
    checkpoint_file: Optional[str] = None,
):

    try:
        if not args.corpus_filename:
            raise ValueError("Please select a corpus file")

        if not args.target_folder:
            raise ValueError("Please choose where to store result")

        if not args.corpus_tag:
            raise ValueError("Please specify output tag")

        # if not os.access(args.target_folder, os.W_OK):
        #    raise PermissionError("Expected write permission to target folder, but was denied")

        if not os.path.isfile(args.corpus_filename):
            raise FileNotFoundError(args.corpus_filename)

        os.makedirs(args.target_folder, exist_ok=True)

        checkpoint_filename: Optional[str] = (
            checkpoint_file or f"{corpus_config.corpus_name}_{POS_CHECKPOINT_FILENAME_POSTFIX}"
        )

        compute_result: co_occurrence.ComputeResult = spaCy_co_occurrence_pipeline(
            corpus_config=corpus_config,
            tokens_transform_opts=args.tokens_transform_opts,
            extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
            context_opts=args.context_opts,
            global_threshold_count=args.count_threshold,
            partition_column=partition_key,
            checkpoint_filename=checkpoint_filename,
        ).value()

        # logger.debug(args.tokens_transform_opts)
        # logger.debug(args.extract_tagged_tokens_opts)
        # logger.debug(args.tagged_tokens_filter_opts.props)
        # logger.debug(args.context_opts)
        # logger.debug(args.count_threshold)
        # logger.debug(args.partition_key)

        if len(compute_result.co_occurrences) == 0:
            raise ValueError("Computation ended up in ZERO records. Check settinsgs!")

        co_occurrence_filename = co_occurrence.folder_and_tag_to_filename(
            folder=args.target_folder, tag=args.corpus_tag
        )

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
                output_filename=co_occurrence_filename,
                partition_keys=[partition_key],
                count_threshold=args.count_threshold,
            ),
        )

        co_occurrence.store_bundle(co_occurrence_filename, bundle)

        (done_callback or default_done_callback)(bundle=bundle)

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
