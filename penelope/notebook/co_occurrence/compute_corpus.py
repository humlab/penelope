import os

import penelope.co_occurrence as co_occurrence
import penelope.notebook.interface as interface
import penelope.pipeline as pipeline
from penelope.corpus import SparvTokenizedCsvCorpus, VectorizedCorpus
from penelope.utility import getLogger

from .. import interface

logger = getLogger('penelope')


def compute_co_occurrence(
    args: interface.ComputeOpts,
):

    try:

        assert args.is_satisfied()

        # if not args.target_filename.endswith(co_occurrence.CO_OCCURRENCE_FILENAME_POSTFIX):
        #     raise ValueError(f"target filename must end with {co_occurrence.CO_OCCURRENCE_FILENAME_POSTFIX}")

        os.makedirs(args.target_folder, exist_ok=True)

        target_filename = co_occurrence.folder_and_tag_to_filename(folder=args.target_folder, tag=args.corpus_tag)

        corpus: SparvTokenizedCsvCorpus = SparvTokenizedCsvCorpus(
            source=args.corpus_filename,
            reader_opts=args.text_reader_opts,
            extract_tokens_opts=args.extract_tagged_tokens_opts,
            tokens_transform_opts=args.tokens_transform_opts,
        )

        token2id = corpus.token2id  # make one pass to create vocabulary and gather token counts

        compute_result: co_occurrence.ComputeResult = co_occurrence.partitioned_corpus_co_occurrence(
            stream=corpus,
            payload=pipeline.PipelinePayload(effective_document_index=corpus.document_index, token2id=token2id),
            context_opts=args.context_opts,
            global_threshold_count=args.count_threshold,
            partition_column=args.partition_keys[0],
        )

        corpus: VectorizedCorpus = co_occurrence.to_vectorized_corpus(
            co_occurrences=compute_result.co_occurrences,
            document_index=compute_result.document_index,
            value_column='value',
        )

        corpus_folder, corpus_tag = co_occurrence.filename_to_folder_and_tag(target_filename)

        bundle = co_occurrence.Bundle(
            corpus=corpus,
            corpus_tag=corpus_tag,
            corpus_folder=corpus_folder,
            co_occurrences=compute_result.co_occurrences,
            document_index=compute_result.document_index,
            compute_options=co_occurrence.create_options_bundle(
                reader_opts=args.text_reader_opts,
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

        return compute_result

    except (
        ValueError,
        FileNotFoundError,
        PermissionError,
    ) as ex:
        logger.error(ex)
    except Exception as ex:
        logger.error(ex)
