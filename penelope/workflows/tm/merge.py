import shutil

import pandas as pd

from penelope import topic_modelling as ntm


def merge_topics_to_clusters(clusers_filename: str, input_folder: str, output_folder: str):
    cluster_mapping: dict[str, list[int]] = (
        pd.read_csv(clusers_filename, sep='\t').groupby('cluster_name')['topic_id'].agg(list).to_dict()
    )

    inferred_topics: ntm.InferredTopicsData = ntm.InferredTopicsData.load(folder=input_folder, slim=True)

    inferred_topics.merge(cluster_mapping)

    inferred_topics.compress()

    shutil.rmtree(output_folder, ignore_errors=True)

    inferred_topics.store(target_folder=output_folder)

    shutil.copy(f'{input_folder}/model_options.json', f'{output_folder}/model_options.json')
    shutil.copy(clusers_filename, f'{output_folder}/clusters.tsv')
