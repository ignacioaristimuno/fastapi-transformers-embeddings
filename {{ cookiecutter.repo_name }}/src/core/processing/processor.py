from typing import List

from src.models.payload import TextsPayload, ResponseTextsPayload
from src.models.embeddings import TextEmbedding
from src.settings import custom_logger


class Processor:
    """
    Class for wrapping the sentence preprocessing and postprocessing for the embedding model.
    """

    def __init__(self, model_id: str, embedding_dim: int) -> None:
        self.logger = custom_logger(self.__class__.__name__)
        self.model_id = model_id
        self.embedding_dim = embedding_dim

    def preprocess_texts(self, payload: TextsPayload) -> TextsPayload:
        """Method for preprocessing the sentences before encoding"""

        if not payload.instruct:
            payload.preprocessed_texts = payload.texts
        else:
            payload.preprocessed_texts = self._add_instruction_prefix(payload.texts)
        return payload

    @staticmethod
    def _add_instruction_prefix(texts: List[str]) -> List[str]:
        """Method for adding an instruction prefix to the texts"""

        instruction_prefix = "Represent this sentence for searching relevant passages:"
        return [f"{instruction_prefix} {text}" for text in texts]

    def postprocess_texts(
        self, payload: TextsPayload, embeddings: list
    ) -> ResponseTextsPayload:
        """Method for postprocessing the embeddings after encoding"""

        return ResponseTextsPayload(
            texts=[
                TextEmbedding(text=text, embedding=embedding)
                for text, embedding in zip(payload.texts, embeddings)
            ],
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
        )
