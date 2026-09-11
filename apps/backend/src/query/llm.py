import hashlib
import json
from dataclasses import dataclass

from openai import OpenAI

from query import config

MAX_REPLY_EXCERPT = 500


@dataclass
class LLMReply:
    content: str
    model: str


def make_client(models_config: dict | None = None) -> OpenAI:
    cfg = models_config if models_config is not None else config.load_models_config()
    return OpenAI(base_url=config.base_url(cfg), api_key=config.api_key())


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


def extract_json_object(content: str) -> dict | None:
    """Find a JSON object in a model reply, tolerating code fences and prose."""
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").strip()
        if text[:4].lower() == "json":
            text = text[4:].lstrip()
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def build_repair_prompt(prompt: str, previous_reply: str) -> str:
    """Ask again after a reply that was not valid JSON with the required fields."""
    excerpt = previous_reply.strip()[:MAX_REPLY_EXCERPT]
    return (
        f"{prompt}\n\nYour previous reply was not valid JSON with the required "
        f"fields:\n{excerpt}\n\nReply with raw JSON only."
    )


def prompt_sha256(system: str, prompt: str) -> str:
    """Hash exactly what decides a verdict, so stale cache entries miss."""
    return hashlib.sha256(f"{system}\n\n{prompt}".encode()).hexdigest()
