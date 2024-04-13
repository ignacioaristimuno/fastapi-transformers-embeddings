from src.core.preprocessing import Preprocessor
from src.models.payload import TextsPayload


def test_preprocessor():
    """Function for testing the Preprocessor class."""
    # Create a preprocessor instance
    preprocessor = Preprocessor()

    # Define some example texts to preprocess
    texts = [
        "I'm feeling so healthy today!",
        "My day was not good, but not bad",
        "I hate when people do this"
    ]
    payload = TextsPayload(texts=texts)

    # Preprocess the texts
    preprocessed_texts: TextsPayload = preprocessor.preprocess_texts(payload)

    # Check the preprocessed texts
    assert len(preprocessed_texts.texts) == len(texts)
