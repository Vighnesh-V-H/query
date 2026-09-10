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


def test_load_models_config_defines_all_roles(monkeypatch):
    monkeypatch.delenv("QUERY_CONFIG_PATH", raising=False)
    cfg = config.load_models_config()
    for role in config.ROLES:
        assert cfg[role]
    assert cfg["provider"]["base_url"].startswith("https://")


def test_model_for_role(models_config, monkeypatch):
    for role in config.ROLES:
        monkeypatch.delenv(f"QUERY_{role.upper()}_MODEL", raising=False)
    assert config.model_for("generator", models_config) == "test/gen-model"
    assert config.model_for("judge", models_config) == "test/judge-model"


def test_model_for_env_override(models_config, monkeypatch):
    monkeypatch.setenv("QUERY_LABELER_MODEL", "test/overridden")
    assert config.model_for("labeler", models_config) == "test/overridden"


def test_model_for_empty_env_override_falls_back(models_config, monkeypatch):
    monkeypatch.setenv("QUERY_LABELER_MODEL", "")
    assert config.model_for("labeler", models_config) == "test/labeler-model"


def test_load_models_config_honors_query_config_path(tmp_path, monkeypatch):
    alt = tmp_path / "alt-models.yaml"
    alt.write_text(
        "generator: alt/gen\n"
        "judge: alt/judge\n"
        "labeler: alt/labeler\n"
        "provider:\n"
        "  base_url: https://alt.invalid/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("QUERY_CONFIG_PATH", str(alt))
    config.load_models_config.cache_clear()
    try:
        cfg = config.load_models_config()
        assert cfg["generator"] == "alt/gen"
        assert cfg["judge"] == "alt/judge"
        assert cfg["labeler"] == "alt/labeler"
        assert config.base_url(cfg) == "https://alt.invalid/v1"
    finally:
        config.load_models_config.cache_clear()


def test_model_for_unknown_role_raises(models_config):
    with pytest.raises(config.ConfigError):
        config.model_for("nonexistent", models_config)


def test_api_key_missing_raises(monkeypatch):
    for name in config.API_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    with pytest.raises(config.ConfigError):
        config.api_key()


def test_api_key_from_env(monkeypatch):
    for name in config.API_KEY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    assert config.api_key() == "sk-test"
