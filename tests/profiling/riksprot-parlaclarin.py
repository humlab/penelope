from penelope import corpus as corpora
from penelope import pipeline, utility
from penelope.pipeline.pipelines import CorpusPipeline

# DATA_FOLDER = "/data/westac/data"
CONFIG_FILENAME = "/data/riksdagen_corpus_data/riksprot-parlaclarin.yml"
OUTPUT_FOLDER = './tests/output'
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.sparv4.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1970.sparv4.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.test.sparv4.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip")

# PYTHONPATH=. topic-model --target-folder ./data --lemmatize --to-lower --min-word-length 1 --only-any-alphanumeric --engine gensim_mallet-lda
# --random-seed 42 --alpha asymmetric --max-iter 3000 --store-corpus /data/riksdagen_corpus_data/riksprot-parlaclarin.yml
# riksprot-parlaclarin-protokoll-50-lemma

CORPUS_FOLDER = '/data/riksdagen_corpus_data/tagged-speech-corpus.numeric.feather'


def run_workflow():
    corpus_config = pipeline.CorpusConfig.load(CONFIG_FILENAME)  # .folders(DATA_FOLDER)
    #    corpus_config.pipeline_payload.files(source=CORPUS_FILENAME, document_index_source=None)
    # corpus_config.checkpoint_opts.deserialize_processes = 3

    transform_opts: corpora.TokensTransformOpts = corpora.TokensTransformOpts(
        to_lower=False,
        to_upper=False,
        min_len=1,
        max_len=None,
        remove_accents=False,
        remove_stopwords=False,
        stopwords=None,
        extra_stopwords=None,
        language='swedish',
        keep_numerals=True,
        keep_symbols=True,
        only_alphabetic=False,
        only_any_alphanumeric=False,
    )
    extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
        pos_includes=None,
        pos_excludes=None,
        pos_paddings=None,
        lemmatize=True,
        append_pos=False,
        global_tf_threshold=1,
        global_tf_threshold_mask=False,
        **corpus_config.pipeline_payload.tagged_columns_names,
    )
    filter_opts: utility.PropertyValueMaskingOpts = utility.PropertyValueMaskingOpts()
    engine_args = {
        'n_topics': 4,
        'passes': 1,
        'random_seed': 42,
        'alpha': 'auto',
        'workers': 1,
        'max_iter': 100,
        'work_folder': './tests/output/',
    }
    extract_opts = "lemma"
    transform_opts = None
    filter_opts = None
    (
        CorpusPipeline(config=corpus_config)
        .load_grouped_id_tagged_frame(
            folder=CORPUS_FOLDER,
            file_pattern='**/prot-*.feather',
            to_tagged_frame=True,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,
            filter_opts=filter_opts,
            transform_opts=transform_opts,
        )
        .to_topic_model(
            corpus_source=None,
            target_folder="./tests/output",
            target_name="APA",
            engine="gensim_lda-multicore",
            engine_args=engine_args,
            store_corpus=True,
            store_compressed=True,
        )
        #.exhaust(n_count=1000)
    )


if __name__ == '__main__':

    run_workflow()
