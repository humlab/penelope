import os
from typing import Any, Dict, List

from penelope.type_alias import PartitionKeys


class PartitionMixIn:
    def partition_documents(self, by: PartitionKeys) -> Dict[Any, List[str]]:

        if 'document_name' not in self.document_index.columns:
            raise ValueError("`document_name` columns missing")

        if isinstance(by, (list, tuple)):
            raise NotImplementedError("multi column partitions is currently not implemented")
            # FIXME: #20 Investigate rule that concatenates concepts
            # by = '_'.join(by)

        groups = self.document_index.groupby(by=by)['document_name'].aggregate(list).to_dict()

        return groups


def stripext(filename):
    return os.path.splitext(filename)[0]
