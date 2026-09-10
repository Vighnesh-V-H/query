import os
from functools import lru_cache

import yaml
from dotenv import load_dotenv

ROLES = ("generator", "judge", "labeler")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "models.yaml")


class ConfigError(Exception):
    pass


@lru_cache(maxsize=1)
def load_models_config() -> dict:
    path = os.environ.get("QUERY_CONFIG_PATH") or MODELS_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"models config at {path} is not a mapping")
    for role in ROLES:
        if not cfg.get(role):
            raise ConfigError(f"models config is missing required role '{role}'")
    provider = cfg.get("provider") or {}
    if not provider.get("base_url"):
        raise ConfigError("models config is missing provider.base_url")
    return dict(cfg)


def model_for(role: str, config: dict | None = None) -> str:
    if role not in ROLES:
        raise ConfigError(f"unknown role '{role}', expected one of {ROLES}")
    cfg = config if config is not None else load_models_config()
    env_key = f"QUERY_{role.upper()}_MODEL"
    override = os.environ.get(env_key)
    if override:
        return override
    return cfg[role]


def base_url(config: dict | None = None) -> str:
    cfg = config if config is not None else load_models_config()
    return cfg["provider"]["base_url"]


def api_key() -> str:
    load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ConfigError("OPENROUTER_API_KEY is not set (see .env.example)")
    return key
