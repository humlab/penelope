# import os

# from penelope.pipeline.config import CorpusConfig
# from penelope.pipeline.spacy.pipelines import spaCy_to_pos_tagged_frame_pipeline

# # def test_compute_pos_statistics():
# partition_key: str = 'year'
# # os.makedirs('./tests/output', exist_ok=True)

# checkpoint_filename: str = "./tests/output/pos_statistics_pos_csv.zip"
# if os.path.isfile(checkpoint_filename):
#     os.remove(checkpoint_filename)

# corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/ssi_corpus_config.yaml').files(
#     source='./tests/test_data/legal_instrument_five_docs_test.zip',
#     index_source='./tests/test_data/legal_instrument_five_docs_test.csv',
#     # source='./tests/test_data/legal_instrument_corpus.zip',
#     # index_source='./tests/test_data/legal_instrument_index.csv',
# )
# pipeline = spaCy_to_pos_tagged_frame_pipeline(
#     corpus_config=corpus_config,
#     checkpoint_filename=checkpoint_filename,
# )

# doc = next(pipeline.resolve())

# # document_year_map = pipeline.payload.document_index.year.to_dict()
# # .project(lambda x: x.content.assign(year=pipeline.payload.document_lookup(x.document_name)['year']))

# # items = [x for x in pipeline.resolve() ]
# # item = items[0]
# # item
# # # pos = compute_pos_statistics(
# # #     document_index=pipeline.payload.document_index,
# # #     df_docs=df_docs,
# # #     group_by_column= "year",
# # #     include_pos = None,
# # # )
# # document_year = pipeline.payload.document_lookup
# # pos_column = pipeline.payload.memory_store.get('pos_column')
# # df_doc.groupby([pos_column]).size().reset_index().rename({0: 'count'}, axis=1)

# # df_doc['year'] = pipeline.payload.document_lookup()

# # pos_statistics = (
# #     pd.concat((df.assign(year=doc_year[i]) for i, df in enumerate(df_docs)))
# #     .groupby(['year', 'pos_'])
# #     .size()
# #     .reset_index()
# #     .rename({0: 'count'}, axis=1)
# #     .assign(count=lambda x: x['count'])
# #     .pivot(index='year', columns="pos_", values='count')
# #     .fillna(0)
# #     .astype(np.int64)
# #     .reset_index()
# # )
