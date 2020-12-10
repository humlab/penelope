import os
from typing import Callable

from penelope.co_occurrence.convert import store_bundle, to_vectorized_corpus
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.spacy.pipelines import spaCy_co_occurrence_pipeline
from penelope.utility import getLogger

from .to_co_occurrence_gui import GUI

logger = getLogger('penelope')
jj = os.path.join

# pylint: disable=unused-argument
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

        checkpoint_filename = f"{corpus_config.corpus_name}_spaCy_pos_tagged_frame_csv.zip"

        co_occurrences = spaCy_co_occurrence_pipeline(
            corpus_config=corpus_config,
            tokens_transform_opts=args.tokens_transform_opts,
            extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
            context_opts=args.context_opts,
            global_threshold_count=args.count_threshold,
            partition_column=partition_key,
            checkpoint_filename=checkpoint_filename,
        ).value()

        co_occurrences.to_csv(
            f"{corpus_config.corpus_name}_{args.corpus_tag}_co-occurrence_csv.zip",
            compression="zip",
            extension='csv',
            header=0,
            sep="\t",
            decimal=',',
            quotechar='"',
        )

        corpus: VectorizedCorpus = to_vectorized_corpus(co_occurrences=co_occurrences, value_column='value_n_t')

        store_bundle(
            jj(args.target_folder, f"{corpus_config.corpus_name}_{args.corpus_tag}_co-occurrence_csv.zip"),
            corpus=corpus,
            corpus_tag=args.corpus_tag,
            input_filename=args.corpus.filename,
            partition_keys=args.partition_keys,
            count_threshold=args.count_threshold,
            co_occurrences=co_occurrences,
            reader_opts=args.reader_opts,
            tokens_transform_opts=args.tokens_transform_opts,
            context_opts=args.context_opts,
            extract_tokens_opts=args.extract_tokens_opts,
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
