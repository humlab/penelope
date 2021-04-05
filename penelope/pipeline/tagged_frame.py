# from collections import defaultdict
# from typing import Any,  Iterator
# from penelope.corpus import DocumentIndex
import pandas as pd

TaggedFrame = pd.core.api.DataFrame
# IdTaggedFrame = pd.core.api.DataFrame

# class TaggedFrameCorpora:

#     def __init__(self, source: Any, document_index: DocumentIndex):
#         self.document_index: DocumentIndex = document_index


# TODO Create combined dictionary both TEXT and LEMMA
# TODO Handle codes for PoS (depending on PoS-schema, might be better to store mapping[int, PoS-tag])
# TODO Function that switches or combines TaggedDocumentFrame =>
# class IdTaggedFrame:
#     """A tagged document represented as a pandas dataframe"""

#     def __init__(
#         self,
#         tagged_frame: TaggedFrame,
#         token2id: Token2Id = None,
#         token_column="token",
#         lemma_column="baseform",
#         pos_column="pos",
#     ):
#         self.columns_names = {
#             "text_column": token_column,
#             "lemma_column": lemma_column,
#             "pos_column": pos_column,
#         }
#         self.tagged_frame = tagged_frame
#         self.id2token = id2token

#         if 'token_id' not in self.tagged_frame.columns:
#             self.tagged_frame['pos_id'] = map(token2id, self.tagged_frame[]


# @dataclass
# class ToIdTaggedFrame(CountTokensMixIn, ITask):
#     """Convert a string based tagged frame to id based tagged frame """
#     attributes: List[str] = None
#     attribute_value_filters: Dict[str, Any] = None

#     def setup(self) -> ITask:
#         self.pipeline.put("tagged_attributes", self.attributes)
#         return self

#     def __post_init__(self):
#         self.in_content_type = [ContentType.TAGGED_FRAME]
#         self.out_content_type = ContentType.TAGGED_ID_FRAME

#     def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

#         tagged_frame: IdTaggedFrame = self.convert(
#             payload=payload,
#             attributes=self.attributes,
#             attribute_value_filters=self.attribute_value_filters,
#         )

#         payload = payload.update(self.out_content_type, tagged_frame)

#         return payload

#     def convert(self, payload):
#         pass
