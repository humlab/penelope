import os

from penelope.co_occurrence import Bundle, ContextOpts
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts
from penelope.pipeline import CorpusConfig, CorpusPipeline
from penelope.pipeline.sparv.pipelines import to_tagged_frame_pipeline
from penelope.utility.filename_utils import strip_path_and_extension

jj = os.path.join


def execute_co_occurrence(corpus_filename: str, output_folder: str):

    os.makedirs('./tests/output', exist_ok=True)

    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/riksdagens-protokoll.yml', corpus_filename)

    context_opts: ContextOpts = ContextOpts(
        concept=set(),
        ignore_concept=False,
        context_width=2,
        pad="*",
    )
    # transform_opts: TokensTransformOpts = TokensTransformOpts()
    extract_opts = ExtractTaggedTokensOpts(
        pos_includes='NN|PM|VB',
        pos_excludes='MAD|MID|PAD',
        pos_paddings="AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO",
        lemmatize=True,
        to_lowercase=True,
    )
    reader_opts = TextReaderOpts(
        filename_fields="year:_:1",
        n_processes=4,
        n_chunksize=50,
    )

    corpus_config.text_reader_opts.update(n_processes=reader_opts.n_processes, n_chunksize=reader_opts.n_chunksize)

    pipeline: CorpusPipeline = (
        to_tagged_frame_pipeline(corpus_config, corpus_filename)
        .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=None, transform_opts=None)
        .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=True)
        .to_corpus_co_occurrence(context_opts=context_opts, global_threshold_count=1)
    )

    bundle: Bundle = pipeline.value()

    basename: str = strip_path_and_extension(corpus_filename)

    bundle.store(folder=output_folder, tag=basename)


if __name__ == "__main__":
    # CORPUS_FILENAME: str = '/data/westac/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip'
    CORPUS_FILENAME: str = '/data/westac/data/riksdagens-protokoll.1920-2019.test.sparv4.csv.zip'
    # CORPUS_FILENAME: str = '/data/westac/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip'
    OUTPUT_FOLDER: str = "tests/output"
    execute_co_occurrence(CORPUS_FILENAME, OUTPUT_FOLDER)
