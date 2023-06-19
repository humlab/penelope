import uuid

from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts
from penelope.pipeline import CorpusConfig, CorpusPipeline, DocumentPayload
from penelope.pipeline.spacy import SpacyTagger

# pylint: disable=redefined-outer-name


def ssi_corpus_config() -> CorpusConfig:
    config: CorpusConfig = CorpusConfig.load('./tests/test_data/SSI/SSI.yml')
    config.pipeline_payload.source = './tests/test_data/SSI/legal_instrument_five_docs_test.zip'
    config.pipeline_payload.document_index_source = './tests/test_data/SSI/legal_instrument_five_docs_test.csv'
    return config


def ssi_topic_model_payload(config: CorpusConfig, tagger: SpacyTagger) -> DocumentPayload:
    target_name: str = f'{uuid.uuid1()}'
    transform_opts: TokensTransformOpts = TokensTransformOpts()
    reader_opts: TextReaderOpts = TextReaderOpts(filename_pattern="*.txt", filename_fields="year:_:1")
    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='|VERB|NOUN|',
        pos_paddings='|ADJ|',
        **config.pipeline_payload.tagged_columns_names,
        filter_opts=dict(is_punct=False),
    )
    target_name: str = {str(uuid.uuid4())[:8]}
    default_engine_args: dict = {
        'n_topics': 4,
        'passes': 1,
        'random_seed': 42,
        'workers': 1,
        'max_iter': 100,
        'work_folder': f'./tests/output/{target_name}',
    }
    transform_opts = None
    payload: DocumentPayload = (
        CorpusPipeline(config=config)
        .load_text(reader_opts=reader_opts, transform_opts=TokensTransformOpts())
        .text_to_spacy(tagger=tagger)
        .to_tagged_frame()
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts)
        .to_topic_model(
            target_mode='both',
            target_folder="./tests/output",
            target_name=target_name,
            engine="gensim_lda-multicore",
            engine_args=default_engine_args,
            store_corpus=True,
            store_compressed=True,
        )
    ).single()
    return payload
