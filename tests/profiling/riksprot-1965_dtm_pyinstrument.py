from penelope import utility
from penelope.scripts.dtm import vectorize_id as workflow

OPTIONS_FILENAME = "tests/profiling/riksprot-1965_dtm_opts.yml"


def run_workflow():

    arguments = utility.update_dict_from_yaml(
        OPTIONS_FILENAME,
        {
            'config_filename': 'tests/profiling/riksprot-1965_corpus_config.yml',
            'corpus_source': '/data/riksdagen_corpus_data/tagged_frames_v0.4.1_speeches.feather',
            'output_folder': './tests/output/',
            'output_tag': 'dtm_1965',
            'filename_pattern': "**/prot-1965*.feather",
        },
    )

    workflow.process(**arguments)


if __name__ == '__main__':

    run_workflow()
