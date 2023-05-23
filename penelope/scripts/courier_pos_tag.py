import os
import re

import click  # pylint: disable=unused-import

from penelope.corpus import TextTransformOpts
from penelope.pipeline import DEFAULT_TAGGED_FRAMES_FILENAME_SUFFIX, CorpusConfig, CorpusPipeline
from penelope.utility import load_cwd_dotenv, path_add_suffix

HYPHEN_REGEXP = re.compile(r'\b(\w+)[-¬]\s*\r?\n\s*(\w+)\s*\b', re.UNICODE)


def remove_hyphens(text: str) -> str:
    result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result


@click.command()
@click.argument('config-filename', type=click.STRING)
def main(config_filename: str = None):  # pylint: disable=redefined-outer-name
    load_cwd_dotenv()

    text_transform_opts = TextTransformOpts(
        fix_hyphenation=True, fix_whitespaces=True, fix_accents=True, extra_transforms=[remove_hyphens]
    )

    config: CorpusConfig = CorpusConfig.load(path=config_filename)

    tagged_corpus_source: str = os.path.abspath(
        path_add_suffix(config.pipeline_payload.source, DEFAULT_TAGGED_FRAMES_FILENAME_SUFFIX)
    )

    pipeline = (
        CorpusPipeline(config=config)
        .load_text(reader_opts=config.text_reader_opts, transform_opts=text_transform_opts)
        .to_tagged_frame(tagger=config.tagger)
        .checkpoint(filename=tagged_corpus_source)
        .checkpoint_feather(folder=config.checkpoint_opts.feather_folder, force=True)
    )

    _ = pipeline.exhaust()


if __name__ == "__main__":
    # config_filename = './opts/inidun/configs/courier_article.yml'
    # main(config_filename=config_filename)
    main()
