import click

from penelope.scripts.utils import option2

from ..pipeline import CorpusConfig


@click.command()
@click.argument('config-filename', type=click.STRING)
@option2('--corpus-source', default=None)
def main(config_filename: str = None, corpus_source: str = None):

    config: CorpusConfig = CorpusConfig.load(path=config_filename)
    corpus_source: str = corpus_source or config.pipeline_payload.source

    _ = config.get_pipeline(
        "tagged_frame_pipeline",
        corpus_source=corpus_source,
        enable_checkpoint=True,
        force_checkpoint=True,
    ).exhaust()


if __name__ == "__main__":
    main()
