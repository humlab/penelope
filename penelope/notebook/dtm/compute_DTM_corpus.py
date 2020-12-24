import os
from typing import Callable

from penelope.pipeline.config import CorpusConfig
from penelope.workflows import vectorize_corpus_workflow

from ..utility import default_done_callback
from .to_DTM_gui import ComputeGUI


# pylint: disable=unused-argument
def compute_document_term_matrix(
    corpus_config: CorpusConfig,
    pipeline_factory: Callable,
    args: ComputeGUI,
    done_callback: Callable,
    persist: bool = False,
):

    try:

        if not args.corpus_filename:
            raise ValueError("please specify corpus file")

        if not args.target_folder:
            raise ValueError("please specify output folder")

        if not os.path.isfile(args.corpus_filename):
            raise FileNotFoundError(args.corpus_filename)

        # FIXME: #23 Addindex field to vectorize workflow (index_field)
        corpus = vectorize_corpus_workflow(
            corpus_type=args.corpus_type.value,
            input_filename=args.corpus_filename,
            output_folder=args.target_folder,
            output_tag=args.target_folder,
            create_subfolder=args.create_subfolder.value,
            filename_field=args.filename_fields.value,
            count_threshold=args.count_threshold.value,
            extract_tokens_opts=args.extract_tagged_tokens_opts,
            tokens_transform_opts=args.tokens_transform_opts,
        )

        (done_callback or default_done_callback)(
            corpus=corpus,
            corpus_tag=args.corpus_tag,
            corpus_folder=args.target_folder,
        )

    except Exception as ex:
        raise ex
