from fastapi import APIRouter, Request
from fastapi import Body

from src.models.payload import TextsPayload, ResponseTextsPayload
from src.settings import custom_logger


logger = custom_logger("Encoding Router")

encoding_router = APIRouter()


@encoding_router.post("/texts")
def encode_texts(
    request: Request, texts: TextsPayload = Body(...)
) -> ResponseTextsPayload:
    """Endpoint for encoding a list of texts."""

    preprocessed_payload: TextsPayload = request.state.processor.preprocess_texts(texts)
    embeddings: list = request.state.embedder.encode_batch(preprocessed_payload)
    response: ResponseTextsPayload = request.state.processor.postprocess_texts(
        texts, embeddings
    )
    return response
