from src.core.encoding import Embedder
from src.models.payload import TextsPayload, ResponseTextsPayload
from src.settings import SettingsManager


def test_classifier():
    """Function for testing the Classifier class."""

    # Load settings
    settings = SettingsManager()

    # Create an embedder instance
    embedder = Embedder(model_id=settings.MODEL_ID, batch_size=settings.BATCH_SIZE)

    # Define the example texts with their expected scores
    example_texts = [
        "I'm feeling so healthy today!",
        "My day was not good, but not bad",
        "I hate when people do this",
    ]

    # Create a payload with the example texts
    input_payload = TextsPayload(texts=example_texts, preprocessed_texts=example_texts)

    # Get predictions
    embeddings: list = embedder.encode_batch(payload=input_payload)

    # Check the number of embeddings
    assert len(embeddings) == len(example_texts)

    # Check the embeddings dimension
    assert len(embeddings[0]) == embedder.model_config.hidden_size
