import collections
import os
import uuid

from penelope import co_occurrence
from penelope.co_occurrence import Bundle, ContextOpts, CoOccurrenceHelper
from penelope.corpus import ITokenizedCorpus, Token2Id, VectorizedCorpus
from penelope.corpus.dtm.vectorizer import CorpusVectorizer
from penelope.corpus.readers.interfaces import ExtractTaggedTokensOpts
from penelope.pipeline.co_occurrence.tasks import CoOccurrenceMatrixBundle, TTM_to_coo_DTM
from penelope.pipeline.config import CorpusConfig
from penelope.pipeline.pipelines import CorpusPipeline
from penelope.utility.pandas_utils import PropertyValueMaskingOpts

from ..fixtures import SIMPLE_CORPUS_ABCDEFG_3DOCS, very_simple_corpus
from ..utils import OUTPUT_FOLDER

jj = os.path.join


# def create_simple_helper() -> CoOccurrenceHelper:
#     return create_bundle_helper(
#         create_bundle_helper(create_simple_bundle()),
#     )

# def create_simple_bundle() -> Bundle:
#     tag: str = "TERRA"
#     folder: str = jj(OUTPUT_FOLDER, tag)
#     simple_corpus = very_simple_corpus(SIMPLE_CORPUS_ABCDEFG_3DOCS)
#     context_opts: co_occurrence.ContextOpts = co_occurrence.ContextOpts(
#         concept={}, ignore_concept=False, context_width=2
#     )
#     bundle: Bundle = create_co_occurrence_bundle(
#         corpus=simple_corpus, context_opts=context_opts, folder=folder, tag=tag
#     )
#     return bundle


# def create_bundle_helper(bundle: Bundle) -> CoOccurrenceHelper:
#     helper: CoOccurrenceHelper = CoOccurrenceHelper(
#         bundle.co_occurrences,
#         bundle.token2id,
#         bundle.document_index,
#     )
#     return helper


def test_create_test_bundle(config: CorpusConfig):

    checkpoint_filename: str = os.path.join(OUTPUT_FOLDER, f'{uuid.uuid1()}_checkpoint_pos_tagged_test.zip')

    extract_opts: ExtractTaggedTokensOpts = ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='|NOUN|', pos_paddings=None
    )
    filter_opts: PropertyValueMaskingOpts = PropertyValueMaskingOpts(is_punct=False)

    corpus: VectorizedCorpus = (
        (
            CorpusPipeline(config=config)
            .tagged_frame_to_tokens(extract_opts=extract_opts, filter_opts=filter_opts, transform_opts=None)
            .to_dtm()
        )
        .single()
        .content
    )

    corpus.dump(tag="kallekulakurtkurt", folder=OUTPUT_FOLDER)
    assert isinstance(corpus, VectorizedCorpus)
    assert corpus.data.shape[0] == 5
    assert len(corpus.token2id) == corpus.data.shape[1]

    os.remove(checkpoint_filename)


# def create_co_occurrence_bundle(
#     *, t_corpus: ITokenizedCorpus, context_opts: ContextOpts, folder: str, tag: str
# ) -> Bundle:

#     stream: Iterable[CoOccurrenceMatrixBundle] = (
#         CoOccurrenceMatrixBundle(
#             document_id,
#             CorpusVectorizer()
#             .fit_transform([doc], already_tokenized=True, vocabulary=t_corpus.token2id)
#             .co_occurrence_matrix(),
#             collections.Counter(),
#         )
#         for document_id, doc in enumerate(t_corpus)
#     )

#     corpus: VectorizedCorpus = TTM_to_coo_DTM(stream, t_token2id, t_document_index)

#     assert corpus is not None

#     token2id: Token2Id = Token2Id(corpus.token2id)

#     # value: CoOccurrenceComputeResult = compute_corpus_co_occurrence(
#     #     stream=corpus,
#     #     document_index=corpus.document_index,
#     #     token2id=token2id,
#     #     context_opts=context_opts,
#     #     global_threshold_count=1,
#     # )

#     # corpus = co_occurrences_to_co_occurrence_corpus(
#     #     co_occurrences=value.co_occurrences,
#     #     document_index=value.document_index,
#     #     token2id=token2id,
#     # )

#     bundle: Bundle = Bundle(
#         folder=folder,
#         tag=tag,
#         co_occurrences=value.co_occurrences,
#         document_index=value.document_index,
#         token_window_counts=value.token_window_counts,
#         token2id=value.token2id,
#         compute_options={},
#         corpus=corpus,
#     )

#     return bundle
