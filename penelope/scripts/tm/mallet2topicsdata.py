from __future__ import annotations

import click

from penelope.vendor.mallet_api.convert import (
    convert_dictionary,
    convert_document_index,
    convert_document_topics,
    convert_overview,
    convert_topic_tokens,
    to_feather,
)

# pylint: disable=unused-argument, too-many-arguments

"""
Convert MALLET topic model result to InferredTopicsData
"""


@click.command()
@click.argument('trained-model-folder', required=True)
def main(trained_model_folder: str):

    convert_document_index(trained_model_folder)
    convert_dictionary(trained_model_folder)
    convert_overview(trained_model_folder)
    convert_topic_tokens(trained_model_folder)
    convert_document_topics(trained_model_folder, normalize=True, epsilon=0.005)
    to_feather(trained_model_folder)


if __name__ == '__main__':

    main()  # pylint: disable=no-value-for-parameter

    from click.testing import CliRunner

    print(CliRunner().invoke(main, ['**/192[0123]/*.feather']).output)
