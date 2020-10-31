# import penelope.vendor.textacy as textacy_utility


# # pylint: disable=too-many-instance-attributes
# class CorpusContainer:
#     """Singleton class for current (last) computed or loaded corpus"""

#     corpus_container = None

#     def __init__(self):
#         self.language = None
#         self.source_path = None
#         self.prepped_source_path = None
#         self.textacy_corpus_path = None
#         self.textacy_corpus = None
#         self.nlp = None
#         self.word_count_scores = None
#         self.document_index = None
#         self.word_count_scores = {}

#     def get_word_count(self, normalize):
#         key = 'word_count_' + normalize
#         self.word_count_scores = self.word_count_scores or {}
#         if key not in self.word_count_scores:
#             self.word_count_scores[key] = textacy_utility.generate_word_count_score(self.textacy_corpus, normalize, 100)
#         return self.word_count_scores[key]

#     def get_word_document_count(self, normalize):
#         key = 'word_document_count_' + normalize
#         self.word_count_scores = self.word_count_scores or {}
#         if key not in self.word_count_scores:
#             self.word_count_scores[key] = textacy_utility.generate_word_document_count_score(
#                 self.textacy_corpus, normalize, 75
#             )
#         return self.word_count_scores[key]

#     @staticmethod
#     def container():

#         CorpusContainer.corpus_container = CorpusContainer.corpus_container or CorpusContainer()

#         return CorpusContainer.corpus_container

#     @staticmethod
#     def corpus():
#         class CorpusNotLoaded(Exception):
#             pass

#         if CorpusContainer.container().textacy_corpus is None:
#             raise CorpusNotLoaded('Corpus not loaded or computed')

#         return CorpusContainer.container().textacy_corpus
