from transformers import AutoConfig

from src.core.processing import Processor
from src.models.payload import TextsPayload
from src.settings import SettingsManager


def test_preprocessor():
    """Function for testing the Preprocessor class."""

    # Load settings
    settings = SettingsManager()

    # Get model embedding dimension
    model_config = AutoConfig.from_pretrained(settings.MODEL_ID)

    # Create a preprocessor instance
    processor = Processor(
        model_id=settings.MODEL_ID,
        embedding_dim=model_config.hidden_size,
    )

    # Define some example texts to preprocess
    texts = [
        "I'm feeling so healthy today!",
        "My day was not good, but not bad",
        "I hate when people do this",
    ]
    payload = TextsPayload(texts=texts)

    # Preprocess the texts
    preprocessed_texts: TextsPayload = processor.preprocess_texts(payload)

    # Check the preprocessed texts
    assert len(preprocessed_texts.texts) == len(texts)
