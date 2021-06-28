from os.path import join as jj

from penelope import corpus as corpora
from penelope import pipeline, utility, workflows
from penelope.co_occurrence import ContextOpts
from penelope.notebook.interface import ComputeOpts

DATA_FOLDER = "./tests/test_data"
CONFIG_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.yml")
OUTPUT_FOLDER = jj(DATA_FOLDER, '../output')
CORPUS_FILENAME = jj(DATA_FOLDER, "riksdagens-protokoll.1920-2019.test.zip")

CONCEPT = set(['information'])  # {'information'}

SUC_SCHEMA: utility.PoS_Tag_Scheme = utility.PoS_Tag_Schemes.SUC
POS_TARGETS: str = 'NN|PM'
POS_PADDINGS: str = 'AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO|VB'
POS_EXLUDES: str = 'MAD|MID|PAD'

corpus_config = pipeline.CorpusConfig.load(CONFIG_FILENAME).folders(DATA_FOLDER)

compute_opts = ComputeOpts(
    corpus_type=pipeline.CorpusType.SparvCSV,
    corpus_filename=CORPUS_FILENAME,
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
        pos_includes=POS_TARGETS,
        pos_excludes=POS_EXLUDES,
        pos_paddings=POS_PADDINGS,
        lemmatize=True,
        to_lowercase=True,
        append_pos=False,
        global_tf_threshold=1,
        global_tf_threshold_mask=False,
    ),
    filter_opts=utility.PropertyValueMaskingOpts(),
    vectorize_opts=corpora.VectorizeOpts(
        already_tokenized=True, lowercase=False, stop_words=None, max_df=1.0, min_df=1, verbose=False
    ),
    tf_threshold=1,
    tf_threshold_mask=False,
    create_subfolder=True,
    persist=True,
    context_opts=ContextOpts(
        context_width=2,
        concept=CONCEPT,
        ignore_concept=False,
        partition_keys=['document_name'],
    ),
    enable_checkpoint=True,
    force_checkpoint=False,
)

corpus_config.pipeline_payload.files(
    source=compute_opts.corpus_filename,
    document_index_source=None,
)

# corpus_config.checkpoint_opts.abort_at_index = 50
# compute_opts.corpus_filename = '/data/westac/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip'


def run_workflow():
    _ = workflows.co_occurrence.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        checkpoint_file=jj(OUTPUT_FOLDER, 'test.zip'),
    )


if __name__ == '__main__':

    import cProfile

    cProfile.run('run_workflow()')
