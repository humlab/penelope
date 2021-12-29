from os.path import join as jj

import penelope.workflows.co_occurrence.compute as workflow
from penelope import corpus as corpora
from penelope import pipeline
from penelope.co_occurrence import ContextOpts
from penelope.workflows.interface import ComputeOpts

# DATA_FOLDER = "./tests/test_data"
# CONFIG_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.yml")
# OUTPUT_FOLDER = jj(DATA_FOLDER, '../output')
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.test.zip")

DATA_FOLDER = "/data/westac/data"
CONFIG_FILENAME = "/home/roger/source/penelope/doit.yml"
OUTPUT_FOLDER = './tests/output'
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.sparv4.csv.zip")
CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1970.sparv4.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.test.sparv4.csv.zip")
# CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip")


def run_workflow():
    corpus_config = pipeline.CorpusConfig.load(CONFIG_FILENAME).folders(DATA_FOLDER)
    corpus_config.pipeline_payload.files(source=CORPUS_FILENAME, document_index_source=None)
    corpus_config.checkpoint_opts.deserialize_processes = 4

    compute_opts = ComputeOpts(
        corpus_type=pipeline.CorpusType.SparvCSV,
        corpus_source=CORPUS_FILENAME,
        target_folder=jj(OUTPUT_FOLDER, 'APA'),
        corpus_tag='APA',
        transform_opts=corpora.TokensTransformOpts(
            to_lower=True,
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
        ),
        text_reader_opts=corpora.TextReaderOpts(
            filename_pattern='*.csv',
            filename_filter=None,
            filename_fields=[
                'year:prot\\_(\\d{4}).*',
                'year2:prot_\\d{4}(\\d{2})__*',
                'number:prot_\\d+[afk_]{0,4}__(\\d+).*',
            ],
            index_field=None,
            as_binary=False,
            sep='\t',
            quoting=3,
        ),
        extract_opts=corpora.ExtractTaggedTokensOpts(
            pos_includes='NN|PM',
            pos_excludes='MAD|MID|PAD',
            pos_paddings='AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO|VB',
            lemmatize=True,
            append_pos=False,
            global_tf_threshold=1,
            global_tf_threshold_mask=False,
            **corpus_config.pipeline_payload.tagged_columns_names,
        ),
        vectorize_opts=corpora.VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            stop_words=None,
            max_df=1.0,
            min_df=1,
            verbose=False,
        ),
        tf_threshold=1,
        tf_threshold_mask=False,
        create_subfolder=True,
        persist=True,
        context_opts=ContextOpts(
            context_width=2,
            concept=set(['kammare']),
            ignore_concept=False,
            partition_keys=['document_name'],
            processes=4,
            chunksize=10,
        ),
        enable_checkpoint=False,
        force_checkpoint=False,
    )

    _ = workflow.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        tagged_corpus_source=jj(OUTPUT_FOLDER, 'test.zip'),
    )


if __name__ == '__main__':

    run_workflow()
