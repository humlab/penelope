import os
from typing import Callable

from penelope.pipeline.config import CorpusConfig
from penelope.utility import getLogger, replace_extension
from penelope.workflows import concept_co_occurrence_workflow

from .to_co_occurrence_gui import GUI

logger = getLogger('penelope')


def compute_co_occurrence(corpus_config: CorpusConfig, args: GUI, partition_key: str, done_callback: Callable):

    try:
        if args.courpus_filename is None:
            raise ValueError("Please select a corpus file")

        if args.target_folder == "":
            raise ValueError("Please choose where to store result")

        if args.corpus_tag == "":
            raise ValueError("Please specify output tag")

        if not os.access(args.target_folder, os.W_OK):
            raise PermissionError("expected write permission to target folder, but was denied")

        if not os.path.isfile(args.courpus_filename):
            raise FileNotFoundError(args.courpus_filename)

        os.makedirs(args.target_folder, exist_ok=True)

        output_filename = os.path.join(
            args.target_folder,
            replace_extension(args.corpus_tag, '.coo_concept_context.csv.zip'),
        )

        co_occurrences = concept_co_occurrence_workflow(
            input_filename=args.corpus_filename,
            output_filename=output_filename,
            context_opts=args.context_opts,
            count_threshold=args.count_threshold,
            partition_keys=[partition_key],
            filename_field=corpus_config.text_reader_opts.filename_fields,
            extract_tokens_opts=args.extract_tokens_opts,
            tokens_transform_opts=args.tokens_transform_opts,
        )

        if done_callback is not None:
            done_callback(
                corpus_folder=args.target_folder,
                corpus_tag=args.corpus_tag,
                concept_co_occurrences=co_occurrences,
                concept_co_occurrences_filename=output_filename,
            )

    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
    ) as ex:
        with args.output:
            logger.error(ex)
    except Exception as ex:
        with args.output:
            logger.error(ex)
