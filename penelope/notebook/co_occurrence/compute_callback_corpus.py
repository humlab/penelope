import os
from typing import Callable, Optional

import pandas as pd
from penelope.pipeline.config import CorpusConfig
from penelope.utility import getLogger, replace_extension
from penelope.workflows import co_occurrence_workflow

from .to_co_occurrence_gui import ComputeGUI

logger = getLogger('penelope')


def compute_co_occurrence(
    corpus_config: CorpusConfig,
    args: ComputeGUI,
    partition_key: str,
    done_callback: Callable,
    checkpoint_file: Optional[str] = None,  # pylint: disable=unused-argument
):

    try:
        if not args.corpus_filename:
            raise ValueError("Please select a corpus file")

        if not args.target_folder:
            raise ValueError("Please choose where to store result")

        if not args.corpus_tag:
            raise ValueError("Please specify output tag")

        # if not os.access(args.target_folder, os.W_OK):
        #     raise PermissionError("expected write permission to target folder, but was denied")

        if not os.path.isfile(args.corpus_filename):
            raise FileNotFoundError(args.corpus_filename)

        os.makedirs(args.target_folder, exist_ok=True)

        output_filename = os.path.join(
            args.target_folder,
            replace_extension(args.corpus_tag, '.co_occurrence.csv.zip'),
        )

        co_occurrences: pd.DataFrame = co_occurrence_workflow(
            corpus_filename=args.corpus_filename,
            target_filename=output_filename,
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
                co_occurrences=co_occurrences,
                co_occurrences_filename=output_filename,
            )

    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
    ) as ex:
        logger.error(ex)
    except Exception as ex:
        logger.error(ex)
