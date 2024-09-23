from __future__ import annotations

import click

from penelope.workflows.tm.merge import merge_topics_to_clusters

"""
Merge topics to clusters
"""


@click.command()
@click.argument('cluster_filename', required=True)
@click.argument('input_folder', required=True)
@click.argument('output_folder', required=True)
def main(cluster_filename: str, input_folder: str, output_folder: str):
    merge_topics_to_clusters(cluster_filename, input_folder, output_folder)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
