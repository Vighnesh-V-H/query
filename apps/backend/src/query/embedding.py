"""Local all-MiniLM-L6-v2 embeddings, computed offline through fastembed/ONNX.

Decision 9 in ``docs/decisions.md`` chooses local MiniLM embeddings for
clustering and retrieval: free, offline, and deterministic, which keeps the
15-minute reproduction promise clear of API dependencies. This module is the
one place that loads the model; clustering (ticket 10) and semantic retrieval
(ticket 16) consume the narrow :class:`Embedder` protocol and stay testable
with fakes.

Embeddings are returned L2-normalized, so cosine similarity is a plain dot
product and Euclidean distance ranks like cosine distance. The model runs
through ONNX Runtime, which is deterministic for a fixed model and input; the
first use downloads the ~90 MB model file, and later runs work offline.
"""

import math
from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when local embeddings cannot be computed."""


class Embedder(Protocol):
    """A text embedder: fixed model name, deterministic unit-length vectors."""

    @property
    def model_name(self) -> str:
        """The embedding model that produced the vectors."""
        ...

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """Embed every text, one L2-normalized vector per input."""
        ...


class MiniLMEmbedder:
    """Lazy ONNX MiniLM embedder; the model loads on first use, not on import."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self._model_name = model_name
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        try:
            model = self._model
            if model is None:
                from fastembed import TextEmbedding  # deferred: loads ONNX Runtime

                model = TextEmbedding(self._model_name)
                self._model = model
            vectors = tuple(
                tuple(float(value) for value in vector) for vector in model.embed(texts)
            )
        except Exception as exc:
            raise EmbeddingError(
                f"failed to embed {len(texts)} texts with {self._model_name}: {exc}"
            ) from exc
        return tuple(l2_normalize(vector) for vector in vectors)


def l2_normalize(vector: Sequence[float]) -> tuple[float, ...]:
    """Scale a vector to unit length; a zero vector is returned unchanged."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(float(value) for value in vector)
    return tuple(value / norm for value in vector)
