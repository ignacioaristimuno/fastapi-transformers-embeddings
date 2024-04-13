from src.core.classification import Classifier
from src.models.payload import TextsPayload, ResponseTextsPayload
from src.settings import SettingsManager


def test_classifier():
    """Function for testing the Classifier class."""

    # Load settings
    settings = SettingsManager()

    # Create a classifier instance
    classifier = Classifier(model_id=settings.MODEL_ID, batch_size=settings.BATCH_SIZE)

    # Define the example texts with their expected scores
    example_texts = [
        {
            "text": "I'm feeling so healthy today!",
            "expected_label": "positive",
            "expected_score_range": (0.95, 1.0),
        },
        {
            "text": "My day was not good, but not bad",
            "expected_label": "negative",
            "expected_score_range": (0.6, 0.7),
        },
        {
            "text": "I hate when people do this",
            "expected_label": "negative",
            "expected_score_range": (0.85, 0.95),
        },
    ]

    # Create a payload with the example texts
    input_payload = TextsPayload(texts=[example["text"] for example in example_texts])

    # Get predictions
    preds: ResponseTextsPayload = classifier.predict(input_payload)

    # Classify the texts and check the scores
    for example, pred in zip(example_texts, preds.texts):
        text = example["text"]
        expected_label = example["expected_label"]
        expected_score_range = example["expected_score_range"]

        # Check the predicted label
        assert pred.label.value == expected_label

        # Check if the score is within the expected range
        assert expected_score_range[0] <= pred.score <= expected_score_range[1]
