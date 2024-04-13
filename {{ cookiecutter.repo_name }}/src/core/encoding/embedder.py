import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List

from src.models.payload import TextsPayload
from src.settings import custom_logger


class Embedder:
    """Class for handling the text encoding operations."""

    def __init__(self, model_id: str, batch_size: int):
        self.logger = custom_logger(self.__class__.__name__)
        self.model_id = model_id
        self.batch_size = batch_size

        # Model loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        self.load_model()
        self._force_complete_loading()

    def load_model(self):
        """Method for loading the model and tokenizer from the Hugging Face model hub."""

        self.logger.info(f"Loading model {self.model_id}")
        self.model_config = AutoConfig.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.logger.info(f"Model {self.model_id} loaded successfully!")

    def _force_complete_loading(self) -> None:
        """Method for avoiding the lazy loading of the model and tokenizer"""

        dummy_texts = ["This is a dummy text", "This is another dummy text"]
        self.encode_batch(
            TextsPayload(texts=dummy_texts, preprocessed_texts=dummy_texts)
        )
        self.logger.info("Finished complete loading of the model and tokenizer!")

    def _create_batches(self, texts: List[str]):
        """Yield successive n-sized chunks from a list of texts."""

        for i in range(0, len(texts), self.batch_size):
            yield texts[i : i + self.batch_size]

    def encode_batch(self, payload: TextsPayload) -> list:
        """Method for encoding a batch of texts."""

        embeddings = []
        for batch_texts in self._create_batches(payload.preprocessed_texts):
            # Model inference
            encoded_inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True
            )
            for k, v in encoded_inputs.items():
                encoded_inputs[k] = v.to(self.device)
            outputs = self.model(**encoded_inputs).last_hidden_state

            # Process the outputs for the entire batch at once
            batch_embeddings = self._pooling(outputs, encoded_inputs, "cls")
            embeddings.append(batch_embeddings)

        embeddings = self._concatenate_embeddings(embeddings)
        return embeddings

    @staticmethod
    def _pooling(
        outputs: torch.Tensor, inputs: Dict, strategy: str = "cls"
    ) -> np.ndarray:
        """Method for pooling the outputs of the model."""
        if strategy == "cls":
            outputs = outputs[:, 0]
        elif strategy == "mean":
            outputs = torch.sum(
                outputs * inputs["attention_mask"][:, :, None], dim=1
            ) / torch.sum(inputs["attention_mask"])
        else:
            raise NotImplementedError
        return outputs.detach().cpu().numpy()

    @staticmethod
    def _concatenate_embeddings(embeddings: list) -> list:
        """Method for concatenating the embeddings of a batch of texts"""

        if isinstance(embeddings[0], ndarray):
            return np.concatenate(embeddings).tolist()
        elif isinstance(embeddings[0], Tensor):
            return torch.cat(embeddings).tolist()
        elif isinstance(embeddings[0], list):
            return embeddings
        else:
            raise ValueError(
                f"Invalid type of embeddings! - Must be ndarray or Tensor and {type(embeddings[0])} found."
            )
