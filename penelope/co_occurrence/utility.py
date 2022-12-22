# from typing import Iterator, List, Mapping, Set

# import numpy as np
# import pandas as pd
# import scipy
# from penelope.corpus import DocumentIndex, Token2Id
# from penelope.type_alias import FilenameTokensTuples
# from penelope.utility import deprecated, strip_extensions
# from scipy.sparse.csr import csr_matrix
# from sklearn.feature_extraction.text import CountVectorizer

# from .bundle import Bundle
# from .convert import term_term_matrix_to_co_occurrences, to_token_window_counts_matrix
# from .interface import ContextOpts, CoOccurrenceError
# from .persistence import WindowCountDTM
# from .vectorize import windows_to_ttm, VectorizedTTM, VectorizeType
# from .windows import generate_windows


# @deprecated
# def compute_non_partitioned_corpus_co_occurrence(
#     stream: FilenameTokensTuples,
#     *,
#     token2id: Token2Id,
#     document_index: DocumentIndex,
#     context_opts: ContextOpts,
#     tf_threshold: int,
#     ingest_tokens: bool = True,
# ) -> Bundle:
#     """Note: This function is currently ONLY used in test cases!"""
#     if token2id is None:
#         raise CoOccurrenceError("expected `token2id` found None")

#     if document_index is None:
#         raise CoOccurrenceError("expected document index found None")

#     if 'n_tokens' not in document_index.columns:
#         raise CoOccurrenceError("expected `document_index.n_tokens`, but found no such column")

#     if 'n_raw_tokens' not in document_index.columns:
#         raise CoOccurrenceError("expected `document_index.n_raw_tokens`, but found no such column")

#     if not isinstance(tf_threshold, int) or tf_threshold < 1:
#         tf_threshold = 1

#     if not isinstance(token2id, Token2Id):
#         token2id = Token2Id(data=token2id)

#     computed_data_frames: List[pd.DataFrame] = []
#     computed_window_counts: Mapping[int, Mapping[int, int]] = dict()

#     vectorizer: DocumentWindowsVectorizer = DocumentWindowsVectorizer(token2id) USE windows_to_ttm!!

#     ignore_ids: Set[int] = None

#     vectorize_type: VectorizeType = VectorizeType.Concept if context_opts.concept else VectorizeType.Normal

#     for filename, tokens in stream:

#         document_id = document_index.loc[strip_extensions(filename)]['document_id']

#         if ingest_tokens:
#             token2id.ingest(tokens)

#         windows = generate_windows(tokens=tokens, context_opts=context_opts)

#         data: Mapping[VectorizeType, VectorizedTTM] = vectorizer.fit_transform(
#             document_id=document_id, windows=windows, context_opts=context_opts
#         )

#         ttm: VectorizedTTM = data.get(vectorize_type)

#         document_co_occurrences = term_term_matrix_to_co_occurrences(
#             ttm.term_term_matrix,
#             threshold_count=1,
#             ignore_ids=ignore_ids,
#         )

#         document_co_occurrences['document_id'] = document_id

#         computed_data_frames.append(document_co_occurrences)
#         computed_window_counts[document_id] = ttm.term_window_counts

#     shape = (len(computed_window_counts), len(token2id))
#     dtm_wc = to_token_window_counts_matrix(computed_window_counts, shape)

#     co_occurrences: pd.DataFrame = pd.concat(computed_data_frames, ignore_index=True)[
#         ['document_id', 'w1_id', 'w2_id', 'value']
#     ]

#     if len(co_occurrences) > 0 and tf_threshold > 1:
#         co_occurrences = co_occurrences[
#             co_occurrences.groupby(["w1_id", "w2_id"])['value'].transform('sum') >= tf_threshold
#         ]

#     return Bundle(
#         co_occurrences=co_occurrences,
#         token2id=token2id,
#         document_index=document_index,
#         window_counts=WindowCountDTM(
#             dtm_wc=dtm_wc,
#         ),
#     )


# @deprecated
# def co_occurrence_matrix(token_ids: Iterator[int], V: int, K: int = 2) -> scipy.sparse.spmatrix:
#     """Computes a sparse co-occurrence matrix given a corpus

#     Source: https://colab.research.google.com/github/henrywoo/MyML/blob/master/Copy_of_nlu_2.ipynb#scrollTo=hPySe-BBEVRy

#     Parameters
#     ----------
#     token_ids : Iterator[int]
#         Corpus token ids
#     V : int
#         A vocabulary size V
#     K : int, optional
#         K (the context window is +-K)., by default 2

#     Returns
#     -------
#     scipy.sparse.spmatrix
#         Sparse co-occurrence matrix
#     """
#     C = scipy.sparse.csc_matrix((V, V), dtype=np.float32)

#     for k in range(1, K + 1):
#         print(f'Counting pairs (i, i | {k}) ...')
#         i = token_ids[:-k]  # current word
#         j = token_ids[k:]  # k words ahead
#         data = (np.ones_like(i), (i, j))  # values, indices
#         Ck_plus = scipy.sparse.coo_matrix(data, shape=C.shape, dtype=np.float32)
#         Ck_plus = scipy.sparse.csc_matrix(Ck_plus)
#         Ck_minus = Ck_plus.T  # consider k words behind
#         C += Ck_plus + Ck_minus

#     print(f"Co-occurrence matrix: {C.shape[0]} words x {C.shape[0]} words")
#     print(f" {C.nnz} nonzero elements")
#     return C


# # https://gist.github.com/zyocum/2ba0457246a4d0075149aa7d607432c1
# # https://www.kaggle.com/ambarish/recommendation-system-donors-choose
# # https://github.com/roothd17/Donor-Choose-ML
# # https://github.com/harrismohammed?tab=repositories


# @deprecated
# class CoOccurrenceCountVectorizer(CountVectorizer):
#     def __init__(self, *, normalize=True, **kwargs):

#         super().__init__(**kwargs)
#         self.normalize = normalize

#     def fit_transform(self, raw_documents, y=None) -> csr_matrix:
#         X = super().fit_transform(raw_documents, y)
#         Xc = X.T * X
#         if self.normalize:
#             g = scipy.sparse.diags(1.0 / Xc.diagonal())
#             Xc = g * Xc
#         else:
#             Xc.setdiag(0)

#         return Xc
