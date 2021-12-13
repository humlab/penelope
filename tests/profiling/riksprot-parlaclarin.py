from penelope import pipeline
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

    # transform_opts: corpora.TokensTransformOpts = corpora.TokensTransformOpts(
    #     to_lower=True,
    #     to_upper=False,
    #     min_len=1,
    #     max_len=None,
    #     remove_accents=False,
    #     remove_stopwords=False,
    #     stopwords=None,
    #     extra_stopwords=None,
    #     language='swedish',
    #     keep_numerals=True,
    #     keep_symbols=True,
    #     only_alphabetic=False,
    #     only_any_alphanumeric=False,
    # )
    # extract_opts: corpora.ExtractTaggedTokensOpts = corpora.ExtractTaggedTokensOpts(
    #     pos_includes='NN|PM',
    #     pos_excludes='MAD|MID|PAD',
    #     pos_paddings='AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO|VB',
    #     lemmatize=True,
    #     append_pos=False,
    #     global_tf_threshold=1,
    #     global_tf_threshold_mask=False,
    #     **corpus_config.pipeline_payload.tagged_columns_names,
    # )
    # filter_opts: utility.PropertyValueMaskingOpts=utility.PropertyValueMaskingOpts()

    (
        CorpusPipeline(config=corpus_config).load_grouped_id_tagged_frame(
            folder=CORPUS_FOLDER,
            file_pattern='**/prot-*.feather',
            to_tagged_frame=True,
        )
        # .tagged_frame_to_tokens(
        #     extract_opts=extract_opts,
        #     filter_opts=filter_opts,
        #     transform_opts=transform_opts,
        # )
        # .to_topic_model(
        #     corpus_source=None,
        #     target_folder="./tests/output",
        #     target_name=target_name,
        #     engine=method,
        #     engine_args=utility.DEFAULT_ENGINE_ARGS,
        #     store_corpus=True,
        #     store_compressed=True,
        # )
        # .vocabulary(
        #     lemmatize=extract_opts.lemmatize,
        #     progress=True,
        #     tf_threshold=extract_opts.global_tf_threshold,
        #     tf_keeps=set(),
        #     close=True,
        # )
        .exhaust(n_count=5000)
        # .tagged_frame_to_tokens(
        #     extract_opts=extract_opts,  # .clear_tf_threshold(),
        #     filter_opts=filter_opts,
        #     transform_opts=transform_opts,
        # ).exhaust()
    )


if __name__ == '__main__':

    run_workflow()
