from dataclasses import dataclass
from typing import Mapping

from penelope.corpus import VectorizedCorpus
from penelope.pipeline import CorpusConfig, id_tagged_frame_to_DTM_pipeline

from .. import compute_opts as copts
from .dtm import store_corpus_bundle

CheckpointPath = str

# pylint: disable=useless-super-delegation


@dataclass
class ComputeOpts(
    copts.ExtractTaggedTokensOptsMixIn,
    copts.TransformOptsMixIn,
    copts.VectorizeOptsMixIn,
    copts.ComputeOptBase,
):
    filename_pattern: str = None

    def is_satisfied(self):
        return super().is_satisfied()

    @property
    def props(self) -> dict:
        return super().props

    def cli_options(self) -> Mapping[str, str]:
        return super().cli_options()

    def ingest(self, data: dict):
        super().ingest(data)
        self.filename_pattern = data.get('filename_pattern')


def compute(args: ComputeOpts, corpus_config: CorpusConfig) -> VectorizedCorpus:

    try:

        assert args.is_satisfied()

        args.extract_opts.global_tf_threshold = 1

        corpus: VectorizedCorpus = id_tagged_frame_to_DTM_pipeline(
            corpus_config=corpus_config,
            corpus_source=args.corpus_source,
            extract_opts=args.extract_opts,
            file_pattern=args.filename_pattern,
            id_to_token=False,
            transform_opts=args.transform_opts,
            vectorize_opts=args.vectorize_opts,
        ).value()

        if (args.tf_threshold or 1) > 1:
            corpus = corpus.slice_by_tf(args.tf_threshold)

        if args.persist:
            store_corpus_bundle(corpus, args)

        return corpus

    except Exception as ex:
        raise ex
