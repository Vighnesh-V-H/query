from dataclasses import dataclass

from openai import OpenAI

from query import config


@dataclass
class LLMReply:
    content: str
    model: str


def make_client(models_config: dict | None = None) -> OpenAI:
    cfg = models_config if models_config is not None else config.load_models_config()
    return OpenAI(base_url=cfg["provider"]["base_url"], api_key=config.api_key())


def call_llm(
    prompt: str,
    role: str = "labeler",
    system: str = "You are a precise, terse assistant.",
    temperature: float = 0.0,
    client: OpenAI | None = None,
    models_config: dict | None = None,
) -> LLMReply:
    cfg = models_config if models_config is not None else config.load_models_config()
    model = config.model_for(role, cfg)
    client = client or make_client(cfg)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    if not response.choices:
        raise RuntimeError(f"model '{model}' returned no choices (upstream error or rate limit)")
    content = response.choices[0].message.content or ""
    return LLMReply(content=content, model=model)
