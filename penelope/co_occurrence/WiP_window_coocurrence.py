
# import os
# from typing import Any, Dict, List

# import pandas as pd

# import penelope.corpus.readers as readers

# from .coocurrence_matrix import compute_coocurrence_matrix

# # THIS FILE COMPUTES COUCCRRENCE FROM PREDEFINED WINDOWS READ FROM EXCEL FILE!

# def load_text_windows(filename: str):
#     """Reads excel file "filename" and returns content as a Pandas DataFrame.
#     The file is written to tsv the first time read for faster subsequent reads.

#     Parameters
#     ----------
#     filename : str
#         Name of excel file that has two columns: year and txt

#     Returns
#     -------
#     [DataFrame]
#         Content of filename as a DataFrame

#     Raises
#     ------
#     FileNotFoundError
#     """
#     filepath = os.path.abspath(filename)

#     if not os.path.isdir(filepath):
#         raise FileNotFoundError("Path {filepath} does not exist!")

#     filebase = os.path.basename(filename).split('.')[0]
#     textfile = os.path.join(filepath, filebase + '.txt')

#     if not os.path.isfile(textfile):
#         df = pd.read_excel(filename)
#         df.to_csv(textfile, sep='\t')

#     df = pd.read_csv(textfile, sep='\t')[['newspaper', 'year', 'txt']]

#     return df


# def compute_for_column_group(df: pd.DataFrame, column_filters: Dict[str,Any], min_count: int, options) -> pd.DataFrame:
#     """[summary]

#     Parameters
#     ----------
#     df : pd.DataFrame
#         [description]
#     column_filters : Dict[str,Any]
#         Dict that specifies (column, value) for each column
#     min_count : int
#         [description]
#     options : [type]
#         [description]

#     Returns
#     -------
#     pd.DataFrame
#         [description]
#     """
#     reader = readers.DataFrameTextTokenizer(df, column_filters=column_filters)

#     df_y = compute_coocurrence_matrix(reader, min_count=min_count, **options)

#     for column, value in column_filters.items():
#         df_y[column] = str(value)

#     return df_y

# def compute_co_ocurrence_for_periods(
#     source_filename: str,
#     column_filters_list: List[Dict[str,Any]],
#     target_filename: str,
#     min_count: int=1,
#     **options
#     ):

#     columns = list(column_filters_list[0].keys()) + ['w1', 'w2', 'value', 'value_n_d', 'value_n_t']

#     df = pd.read_csv(source_filename, sep='\t')[['newspaper', 'year', 'txt']]
#     df_r = pd.DataFrame(columns=columns)

#     n_documents = 0
#     for column_filters in column_filters_list:

#         reader = readers.DataFrameTextTokenizer(df, column_filters=column_filters)
#         df_y = compute_coocurrence_matrix(reader, min_count=min_count, **options)
#         for column, value in column_filters.items():
#             df_y[column] = str(value)
#         df_r = df_r.append(df_y[columns], ignore_index=True)
#         n_documents += len(df_y)

#     print("Done! Processed {} rows...".format(n_documents))

#     # Scale a normalized data matrix to the [0, 1] range:
#     df_r['value_n_t'] = df_r.value_n_t / df_r.value_n_t.max()
#     df_r['value_n_d'] = df_r.value_n_d / df_r.value_n_d.max()

#     extension = target_filename.split(".")[-1]
#     if extension == ".xlsx":
#         df_r.to_excel(target_filename, index=False)
#     elif extension in ["zip", "gzip"]:
#         df_r.to_csv(target_filename, sep='\t', compression=extension, index=False, header=True)
#     else:
#         df_r.to_csv(target_filename, sep='\t', index=False, header=True)


# def compute_co_ocurrence_for_periods(source_filename: str, newspapers, periods, target_filename, min_count=1, **options):

#     columns_filters = [
#         {
#             'newspaper': newspaper,
#             'period':  period
#         } for newspaper in newspapers
#             for period in periods
#     ]

#     n_documents = 0
#     for column_filters in columns_filters:
#         for period in periods:

#             print("Processing: {} {}...".format(newspaper, period))
#             column_filters = {
#                 'newspaper': newspaper,
#                 'year':  period
#             }
#             reader = readers.DataFrameTextTokenizer(df, column_filters=column_filters)
#             df_y = compute_coocurrence_matrix(reader, min_count=min_count, **options)
#             for column, value in column_filters.items():
#                 df_y[column] = str(value)
#             df_r = df_r.append(df_y[columns], ignore_index=True)
#             n_documents += len(df_y)

#     print("Done! Processed {} rows...".format(n_documents))

#     # Scale a normalized data matrix to the [0, 1] range:
#     df_r['value_n_t'] = df_r.value_n_t / df_r.value_n_t.max()
#     df_r['value_n_d'] = df_r.value_n_d / df_r.value_n_d.max()

#     extension = target_filename.split(".")[-1]
#     if extension == ".xlsx":
#         df_r.to_excel(target_filename, index=False)
#     elif extension in ["zip", "gzip"]:
#         df_r.to_csv(target_filename, sep='\t', compression=extension, index=False, header=True)
#     else:
#         df_r.to_csv(target_filename, sep='\t', index=False, header=True)

