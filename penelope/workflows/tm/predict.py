import sys

import click

import penelope.corpus as penelope
from penelope import pipeline


def compute(
    *,
    config: pipeline.CorpusConfig,
    model_folder: str = None,
    model_name: str = None,
    target_folder: str = None,
    target_name: str,
    corpus_source: str = None,
    extract_opts: penelope.ExtractTaggedTokensOpts = None,
    transform_opts: penelope.TokensTransformOpts = None,
    minimum_probability: float = 0.001,
    n_tokens: int = 200,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
) -> dict:

    corpus_source: str = corpus_source or config.pipeline_payload.source

    if corpus_source is None:
        click.echo("usage: either corpus-folder or corpus filename must be specified")
        sys.exit(1)

    tag, folder = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
        )
        .predict_topics(
            model_folder=model_folder,
            model_name=model_name,
            target_folder=target_folder,
            target_name=target_name,
            minimum_probability=minimum_probability,
            n_tokens=n_tokens,
        )
    ).value()

    return (tag, folder)
