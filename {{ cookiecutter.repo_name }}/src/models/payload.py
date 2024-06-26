from pydantic import BaseModel
from typing import List

from src.models.embeddings import TextEmbedding


class TextsPayload(BaseModel):
    """Input payload for the classification endpoint"""

    texts: List[str]
    instruct: bool = False
    preprocessed_texts: List[str] = None


class ResponseTextsPayload(BaseModel):
    """Response payload for the classification endpoint"""

    texts: List[TextEmbedding]
    model_id: str
    embedding_dim: int
