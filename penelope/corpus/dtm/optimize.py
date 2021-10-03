# from typing import Iterable, Tuple

# import numpy as np
# import scipy
# from numba import jit


# @jit
# def numba_to_term_term_matrix_stream(
#     bag_term_matrix: scipy.sparse.spmatrix, token2pairs: dict, vocab_size: int
# ) -> Tuple[int, Iterable[scipy.sparse.spmatrix]]:
#     """Generates a sequence of term-term matrices for each document (row)"""
#     """Reconstruct ttm row by row"""
#     for i in range(0, bag_term_matrix.shape[0]):
#         document: scipy.sparse.spmatrix = bag_term_matrix[i, :]
#         if len(document.data) == 0:
#             yield i, None
#         else:
#             xy_ids = [token2pairs[i] for i in document.indices]
#             rows = np.array([x[0] for x in xy_ids], dtype=np.int32)
#             cols = np.array([x[1] for x in xy_ids], dtype=np.int32)
#             # rows, cols = zip(*(token2pairs[i] for i in document.indices))
#             term_term_matrix: scipy.sparse.spmatrix = scipy.sparse.csc_matrix(
#                 (document.data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=bag_term_matrix.dtype
#             )
#             yield i, term_term_matrix
