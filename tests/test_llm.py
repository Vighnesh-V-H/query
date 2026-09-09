from query import llm


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, model, messages, temperature):
        self.last_model = model
        self.last_messages = messages
        self.last_temperature = temperature

        class _Msg:
            content = self._content

        class _Choice:
            message = _Msg()

        class _Response:
            choices = [_Choice()]

        return _Response()


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, content="ok"):
        self.completions = _FakeCompletions(content)
        self.chat = _FakeChat(self.completions)


def test_call_llm_uses_configured_role_model():
    fake = _FakeClient(content="ok")
    cfg = {
        "generator": "test/gen",
        "judge": "test/judge",
        "labeler": "test/labeler",
        "provider": {"base_url": "https://example.invalid/v1"},
    }
    reply = llm.call_llm("hi", role="generator", client=fake, models_config=cfg)
    assert reply.content == "ok"
    assert reply.model == "test/gen"
    assert fake.completions.last_model == "test/gen"
    assert fake.completions.last_messages[1]["content"] == "hi"


def test_make_client_uses_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(llm.config, "load_dotenv", lambda: None)
    cfg = {"provider": {"base_url": "https://example.invalid/v1"}}
    client = llm.make_client(cfg)
    assert str(client.base_url).rstrip("/") == "https://example.invalid/v1"
