from os.path import join as jj

from penelope import pipeline
from penelope.utility import find_data_folder

DATA_FOLDER = "/data/westac/riksdagen_corpus_data/"
CONFIG_FILENAME = "/data/westac/riksdagen_corpus_data/riksprot-parlaclarin.yml"
OUTPUT_FOLDER = './tests/output'
CORPUS_FILENAME = jj(DATA_FOLDER, "riksprot_parlaclarin_speech_sequence.stanza.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1970.sparv4.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.test.sparv4.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip")


def run_workflow():
    data_folder: str = find_data_folder(project_name="welfare_state_analytics", project_short_name="westac")
    corpus_folder: str = jj(data_folder, "riksdagen_corpus_data")
    config_filename: str = jj(corpus_folder, "riksprot-parlaclarin.yml")
    corpus_config = pipeline.CorpusConfig.load(config_filename).folders(corpus_folder)
    corpus_config.pipeline_payload.files(source=CORPUS_FILENAME, document_index_source=None)
    corpus_config.serialize_opts.deserialize_processes = 3
    corpus_config.serialize_opts.feather_folder = None

    (
        corpus_config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=CORPUS_FILENAME,
            enable_checkpoint=False,
            force_checkpoint=False,
            text_transform_opts=None,
            update_token_counts=False,
            stop_at_index=10,
        )
        # .vocabulary(
        #     lemmatize=extract_opts.lemmatize,
        #     progress=True,
        #     tf_threshold=extract_opts.global_tf_threshold,
        #     tf_keeps=set(),
        #     close=True,
        # )
        .tqdm().exhaust()
        # .tagged_frame_to_tokens(
        #     extract_opts=extract_opts,  # .clear_tf_threshold(),
        #     transform_opts=transform_opts,
        # ).exhaust()
    )


if __name__ == '__main__':
    run_workflow()
