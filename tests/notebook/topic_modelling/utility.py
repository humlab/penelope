from penelope.topic_modelling import InferredTopicsData

INFERRED_TOPICS_DATA_FOLDER = './tests/test_data/tranströmer_inferred_model'


def load_inferred_topics_data() -> InferredTopicsData:
    inferred_data: InferredTopicsData = InferredTopicsData.load(
        folder=INFERRED_TOPICS_DATA_FOLDER, filename_fields="year:_:1"
    )
    return inferred_data
