import os
from typing import Callable

from penelope.corpus import TokensTransformOpts, VectorizedCorpus
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.pipelines import CorpusPipeline
from penelope.utility import get_logger, path_add_suffix

from ..utility import default_done_callback
from .to_DTM_gui import GUI

logger = get_logger()


def compute_document_term_matrix(
    corpus_config: CorpusConfig,
    pipeline_factory: Callable,
    args: GUI,
    done_callback: Callable,
    persist: bool = False,
):
    try:

        if not args.corpus_tag:
            raise ValueError("please specify CORPUS TAG")

        if not args.target_folder:
            raise ValueError("please specify OUTPUT FOLDER")

        if not os.path.isfile(args.corpus_filename):
            raise FileNotFoundError(f'file {args.corpus_filename} not found')

        checkpoint_filename: str = path_add_suffix(args.corpus_filename, '_pos_csv')

        pipeline: CorpusPipeline = pipeline_factory(
            corpus_config=corpus_config,
            tokens_transform_opts=args.tokens_transform_opts,
            extract_tagged_tokens_opts=args.extract_tagged_tokens_opts,
            tagged_tokens_filter_opts=args.tagged_tokens_filter_opts,
            vectorize_opts=args.vectorize_opts,
            checkpoint_filename=checkpoint_filename,
        )

        corpus = resolve_DTM_pipeline(
            pipeline=pipeline,
            corpus_folder=args.corpus_folder,
            corpus_filename=args.corpus_filename,
            persist=persist,
            corpus_tag=args.corpus_tag,
            target_folder=args.target_folder,
        )

        (done_callback or default_done_callback)(
            corpus=corpus,
            corpus_tag=args.corpus_tag,
            corpus_folder=args.target_folder if persist else None,
        )

    except Exception as ex:
        raise ex


def resolve_DTM_pipeline(
    pipeline: TokensTransformOpts = None,
    corpus_folder: str = None,
    corpus_filename: str = None,
    corpus_tag: str = None,
    target_folder: str = None,
    persist: bool = False,
):
    try:
        if not corpus_filename:
            raise ValueError("corpus file is falsy")

        if not corpus_folder:
            raise ValueError("corpus folder is falsy")

        if persist:

            if not corpus_tag:
                raise ValueError("undefined corpus tag")

            if not target_folder:
                raise ValueError("target folder undefined")

        if not os.path.isfile(corpus_filename):
            raise FileNotFoundError(corpus_filename)

        corpus: VectorizedCorpus = pipeline.value()

        if persist:
            os.makedirs(target_folder, exist_ok=True)
            corpus.dump(tag=corpus_tag, folder=target_folder)

        return corpus

    except Exception as ex:
        raise ex
