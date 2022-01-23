from __future__ import annotations

import sqlite3
import typing as t
from os.path import join as jj

import numpy as np
import pandas as pd

if t.TYPE_CHECKING:
    from ..interfaces import InferredTopicsData


class InferredTopicsDB:
    def __init__(self, folder: str, inferred_topics: InferredTopicsData):

        self.folder: str = folder
        self.inferred_topics: InferredTopicsData = inferred_topics
        self.data: pd.DataFrame = self.inferred_topics.document_topic_weights

    def to_sqlite(self, folder: str, data: InferredTopicsData) -> None:

        """Create an sqlite cache for data"""
        db = sqlite3.connect(jj(folder, "inferred_topics_data.sqlite"))

        # Load the CSV in chunks:
        for c in pd.read_csv("voters.csv", chunksize=1000):
            # Append all rows to a new database table, which
            # we name 'voters':
            c.to_sql("voters", db, if_exists="append")
        # Add an index on the 'street' column:
        db.execute("CREATE INDEX street ON voters(street)")
        db.close()

        with sqlite3.connect(jj(folder, "inferred_topics_data.sqlite")) as db:

            data.dictionary.to_sql("dictionary", db, if_exists="replace")
            data.document_index.to_sql("document_index", db, if_exists="replace")
            data.document_topic_weights.to_sql("document_topic_weights", db, if_exists="replace")
            data.topic_token_overview.to_sql("topic_token_overview", db, if_exists="replace")
            data.topic_token_weights.to_sql("topic_token_weights", db, if_exists="replace")
