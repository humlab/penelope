import os
from typing import Callable, Optional

import pandas as pd
from penelope.co_occurrence import CO_OCCURRENCE_FILENAME_POSTFIX, store_bundle, to_vectorized_corpus
from penelope.corpus import VectorizedCorpus
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility import getLogger
from penelope.utility.file_utility import pandas_to_csv_zip

from ..utility import default_done_callback
from .to_co_occurrence_gui import GUI

logger = getLogger('penelope')
jj = os.path.join

POS_CHECKPOINT_FILENAME_POSTFIX = '_spaCy_pos_tagged_frame_csv.zip'


# pylint: disable=unused-argument
def compute_co_occurrence(
    corpus_config: CorpusConfig,
    args: GUI,
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

        co_occurrences: pd.DataFrame = spaCy_co_occurrence_pipeline(
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

        if len(co_occurrences) == 0:
            raise ValueError("Computation ended up in ZERO records. Check settinsgs!")

        co_occurrence_filename = jj(args.target_folder, f"{args.corpus_tag}{CO_OCCURRENCE_FILENAME_POSTFIX}")

        pandas_to_csv_zip(
            zip_filename=co_occurrence_filename,
            dfs=(co_occurrences, 'co_occurrence.csv'),
            extension='csv',
            header=True,
            sep="\t",
            decimal=',',
            quotechar='"',
        )

        corpus: VectorizedCorpus = to_vectorized_corpus(co_occurrences=co_occurrences, value_column='value_n_t')

        store_bundle(
            co_occurrence_filename,
            corpus=corpus,
            corpus_tag=args.corpus_tag,
            input_filename=args.corpus_filename,
            partition_keys=[partition_key],
            count_threshold=args.count_threshold,
            co_occurrences=co_occurrences,
            reader_opts=corpus_config.text_reader_opts,
            tokens_transform_opts=args.tokens_transform_opts,
            context_opts=args.context_opts,
            extract_tokens_opts=args.extract_tagged_tokens_opts,
        )

        (done_callback or default_done_callback)(
            corpus=corpus,
            corpus_tag=args.corpus_tag,
            corpus_folder=args.target_folder,
            co_occurrences=co_occurrences,
            compute_options=None,
        )

    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
    ) as ex:
        logger.error(ex)
    except Exception as ex:
        logger.error(ex)
