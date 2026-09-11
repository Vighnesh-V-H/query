from query import embedding


def test_l2_normalize_scales_to_unit_length():
    assert embedding.l2_normalize([3.0, 4.0]) == (0.6, 0.8)


def test_l2_normalize_keeps_zero_vector():
    assert embedding.l2_normalize([0.0, 0.0]) == (0.0, 0.0)


def test_embed_with_no_texts_never_loads_the_model():
    embedder = embedding.MiniLMEmbedder()

    assert embedder.embed([]) == ()
    assert embedder._model is None


def test_embed_delegates_to_fastembed_and_normalizes(monkeypatch):
    seen = {}

    class FakeTextEmbedding:
        def __init__(self, model_name):
            seen["model_name"] = model_name

        def embed(self, texts):
            seen["texts"] = list(texts)
            return [[3.0, 4.0] for _ in texts]

    monkeypatch.setattr("fastembed.TextEmbedding", FakeTextEmbedding)

    embedder = embedding.MiniLMEmbedder()
    vectors = embedder.embed(["hello", "world"])

    assert embedder.model_name == embedding.DEFAULT_MODEL_NAME
    assert seen["model_name"] == embedding.DEFAULT_MODEL_NAME
    assert seen["texts"] == ["hello", "world"]
    assert vectors == ((0.6, 0.8), (0.6, 0.8))


def test_embed_wraps_model_errors(monkeypatch):
    class BrokenTextEmbedding:
        def __init__(self, model_name):
            pass

        def embed(self, texts):
            raise RuntimeError("boom")

    monkeypatch.setattr("fastembed.TextEmbedding", BrokenTextEmbedding)

    try:
        embedding.MiniLMEmbedder().embed(["hello"])
    except embedding.EmbeddingError as exc:
        assert "failed to embed 1 texts" in str(exc)
        assert "boom" in str(exc)
    else:
        raise AssertionError("expected EmbeddingError")
