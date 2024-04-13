from pydantic import BaseModel
from typing import List


class TextEmbedding(BaseModel):
    """Text embedding representation"""

    text: str
    embedding: List[float]
