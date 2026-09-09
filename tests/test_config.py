import os

import pytest

from query import config


@pytest.fixture
def models_config():
    return {
        "generator": "test/gen-model",
        "judge": "test/judge-model",
        "labeler": "test/labeler-model",
        "provider": {"base_url": "https://example.invalid/v1"},
    }


def test_load_models_config_defines_all_roles():
    cfg = config.load_models_config()
    for role in config.ROLES:
        assert cfg[role]
    assert cfg["provider"]["base_url"].startswith("https://")


def test_model_for_role(models_config):
    assert config.model_for("generator", models_config) == "test/gen-model"
    assert config.model_for("judge", models_config) == "test/judge-model"


def test_model_for_env_override(models_config, monkeypatch):
    monkeypatch.setenv("QUERY_LABELER_MODEL", "test/overridden")
    assert config.model_for("labeler", models_config) == "test/overridden"


def test_model_for_unknown_role_raises(models_config):
    with pytest.raises(config.ConfigError):
        config.model_for("nonexistent", models_config)


def test_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    with pytest.raises(config.ConfigError):
        config.api_key()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    assert config.api_key() == "sk-test"
