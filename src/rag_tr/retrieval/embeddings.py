import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {text}" for text in texts]
        return self._model.encode(prefixed, normalize_embeddings=True)

    def encode_query(self, text: str) -> np.ndarray:
        embeddings = self._model.encode([f"query: {text}"], normalize_embeddings=True)
        return embeddings[0]
