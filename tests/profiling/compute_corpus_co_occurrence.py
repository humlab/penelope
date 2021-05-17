import os

from penelope.co_occurrence import ContextOpts, CoOccurrenceComputeResult, store_co_occurrences
from penelope.corpus import DocumentIndexHelper, ExtractTaggedTokensOpts, TextReaderOpts
from penelope.pipeline import CorpusConfig, CorpusPipeline
from penelope.pipeline.sparv.pipelines import to_tagged_frame_pipeline
from penelope.utility.filename_utils import strip_path_and_extension

jj = os.path.join


def execute_co_occurrence(corpus_filename: str, output_folder: str):

    os.makedirs('./tests/output', exist_ok=True)

    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/riksdagens-protokoll.yml', corpus_filename)

    context_opts: ContextOpts = ContextOpts(
        concept={},
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

    # stream = SparvCsvTokenizer(corpus_filename, extract_tokens_opts=extract_opts, reader_opts=reader_opts)
    # document_index = stream.document_index
    # token2id: Token2Id = Token2Id()
    # for _, tokens in tqdm.tqdm(stream):
    #     token2id.ingest(tokens)

    pipeline: CorpusPipeline = (
        to_tagged_frame_pipeline(corpus_config, corpus_filename).tagged_frame_to_tokens(
            extract_opts=extract_opts, filter_opts=None
        )
        # .tokens_transform(tokens_transform_opts=transform_opts)
        .to_document_co_occurrence(context_opts=context_opts, ingest_tokens=True)
        # .tqdm()
        .to_corpus_document_co_occurrence(context_opts=context_opts, global_threshold_count=1, ignore_pad=False)
    )

    value: CoOccurrenceComputeResult = pipeline.value()
    basename: str = strip_path_and_extension(corpus_filename)
    store_co_occurrences(jj(output_folder, f"{basename}.co-occurrences.zip"), value.co_occurrences)

    value.token2id.store(
        jj(output_folder, f"{basename}.co-occurrences.dictionary.zip"),
    )

    DocumentIndexHelper(value.document_index).store(
        jj(output_folder, f"{basename}.co-occurrences.document_index.zip"),
    )


if __name__ == "__main__":
    # CORPUS_FILENAME: str = '/data/westac/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip'
    CORPUS_FILENAME: str = '/data/westac/data/riksdagens-protokoll.1920-2019.test.sparv4.csv.zip'
    # CORPUS_FILENAME: str = '/data/westac/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip'
    OUTPUT_FOLDER: str = "tests/output"
    execute_co_occurrence(CORPUS_FILENAME, OUTPUT_FOLDER)
