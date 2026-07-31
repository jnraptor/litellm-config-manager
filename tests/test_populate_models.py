#!/usr/bin/env python3
"""Tests for populate_models.py and ModelMappingLoader save() helpers."""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup_base import ModelMappingLoader, ProviderConfigLoader
from populate_models import (
    ModelsPopulator,
    _normalize_for_match,
    _strip_trailing_suffixes,
    _strip_vendor_prefixes,
    filter_free_models,
    find_model_in_api,
)


@pytest.fixture(autouse=True)
def _reset_provider_singleton():
    """Reset the ProviderConfigLoader singleton around populator tests."""
    ProviderConfigLoader.reset()
    yield
    ProviderConfigLoader.reset()


# ==============================================================================
# Normalization / fuzzy matching
# ==============================================================================


class TestNormalizeForMatch:
    def test_strips_separators(self):
        assert _normalize_for_match("glm-5.1") == "glm51"
        assert _normalize_for_match("glm-5-1") == "glm51"
        assert _normalize_for_match("glm_5.1") == "glm51"

    def test_strips_vendor_prefix(self):
        assert _normalize_for_match("z-ai/glm-5.1") == "glm51"
        assert _normalize_for_match("accounts/fireworks/models/glm-5p1") == "glm51"
        assert _normalize_for_match("anthropic/claude-3-opus") == "claude3opus"

    def test_strips_trailing_suffix(self):
        assert _normalize_for_match("glm-5.1-fw") == "glm51"
        assert _normalize_for_match("kimi-k2.5-fw") == "kimik25"
        assert _normalize_for_match("qwen3.7-max-t") == "qwen37max"
        assert _normalize_for_match("model:free") == "model"
        assert _normalize_for_match("model:latest") == "model"

    def test_case_insensitive(self):
        assert _normalize_for_match("Claude-3-Opus") == "claude3opus"
        assert _normalize_for_match("MINIMAX-M3") == "minimaxm3"

    def test_p_notation_treated_as_separator(self):
        assert _normalize_for_match("glm-5p1") == _normalize_for_match("glm-5.1")
        assert _normalize_for_match("kimi-k2p5") == _normalize_for_match("kimi-k2.5")


class TestStripHelpers:
    def test_strip_vendor_prefixes(self):
        assert _strip_vendor_prefixes("z-ai/glm-5.1") == "glm-5.1"
        assert _strip_vendor_prefixes("accounts/fireworks/models/glm-5.1") == "glm-5.1"
        assert _strip_vendor_prefixes("minimax/minimax-m3") == "minimax-m3"
        assert _strip_vendor_prefixes("glm-5.1") == "glm-5.1"

    def test_strip_trailing_suffixes(self):
        assert _strip_trailing_suffixes("glm-5.1-fw") == "glm-5.1"
        assert _strip_trailing_suffixes("qwen3.7-max-t") == "qwen3.7-max"
        assert _strip_trailing_suffixes("model:free") == "model"
        assert _strip_trailing_suffixes("model") == "model"


class TestFindModelInApi:
    def test_exact_match(self):
        api = {"minimax-m3": {"id": "minimax-m3"}}
        matched, score, _ = find_model_in_api("minimax-m3", api)
        assert matched == "minimax-m3"
        assert score == 1.0

    def test_vendor_prefix_match(self):
        api = {"minimax/minimax-m3": {"id": "minimax/minimax-m3"}}
        matched, score, _ = find_model_in_api("minimax-m3", api)
        assert matched == "minimax/minimax-m3"
        assert score >= 0.85

    def test_normalized_match_dash_vs_dot(self):
        api = {"glm-5.1": {"id": "glm-5.1"}}
        matched, score, _ = find_model_in_api("glm5.1", api)
        assert matched == "glm-5.1"
        assert score >= 0.85

    def test_normalized_match_p1_variant(self):
        api = {"glm-5p1": {"id": "glm-5p1"}}
        matched, score, _ = find_model_in_api("glm-5.1", api)
        assert matched == "glm-5p1"
        assert score >= 0.75

    def test_normalized_match_dash_variant(self):
        api = {"glm-5-1": {"id": "glm-5-1"}}
        matched, score, _ = find_model_in_api("glm-5.1", api)
        assert matched == "glm-5-1"
        assert score >= 0.75

    def test_fireworks_style_match(self):
        api = {"accounts/fireworks/models/glm-5p1": {"id": "x"}}
        matched, score, _ = find_model_in_api("glm-5.1", api)
        assert matched == "accounts/fireworks/models/glm-5p1"
        assert score >= 0.75

    def test_no_match(self):
        api = {"some-other-model": {"id": "x"}}
        matched, score, _ = find_model_in_api("minimax-m3", api)
        assert matched is None
        assert score == 0.0

    def test_empty_api(self):
        assert find_model_in_api("minimax-m3", {}) == (None, 0.0, "no api models")

    def test_prefers_exact_over_substring(self):
        api = {
            "minimax-m3-free": {"id": "a"},
            "minimax-m3": {"id": "b"},
        }
        matched, score, _ = find_model_in_api("minimax-m3", api)
        assert matched == "minimax-m3"
        assert score == 1.0

    def test_no_substring_match_for_short_key(self):
        api = {"some-extremely-long-model-name-free": {"id": "x"}}
        matched, _, _ = find_model_in_api("gpt", api)
        assert matched is None

    def test_substring_match_prefers_shortest_api_id(self):
        api = {
            "minimax-m3-ultra-experimental": {"id": "x"},
            "minimax-m3": {"id": "y"},
        }
        matched, _, _ = find_model_in_api("minimax-m3", api)
        assert matched == "minimax-m3"

    def test_no_false_positive_for_shorter_version(self):
        """A key like kimi-k2.7 should not match the older kimi-k2 variants."""
        api = {
            "kimi-k2": {"id": "kimi-k2"},
            "kimi-k2:1t": {"id": "kimi-k2:1t"},
            "moonshotai/kimi-k2": {"id": "moonshotai/kimi-k2"},
        }
        matched, score, _ = find_model_in_api("kimi-k2.7", api)
        assert matched is None
        assert score == 0.0

    def test_substring_match_allows_longer_variant(self):
        """kimi-k2.7 may match a longer id like kimi-k2.7-code when exact is missing."""
        api = {
            "kimi-k2.7-code": {"id": "kimi-k2.7-code"},
            "kimi-k2": {"id": "kimi-k2"},
        }
        matched, score, _ = find_model_in_api("kimi-k2.7", api)
        assert matched == "kimi-k2.7-code"
        assert score == 0.6


class TestFilterFreeModels:
    def test_keeps_zero_cost_models(self):
        models = {
            "free": {"input_cost": 0.0, "output_cost": 0.0},
            "paid": {"input_cost": 0.1, "output_cost": 0.2},
        }

        assert list(filter_free_models(models)) == ["free"]

    def test_keeps_nominal_free_cost_from_models_dev(self):
        models = {
            "free": {"input_cost": 1.0e-09, "output_cost": 1.0e-09},
            "paid": {"input_cost": 1.0e-09, "output_cost": 2.0e-09},
        }

        assert list(filter_free_models(models)) == ["free"]

    def test_excludes_missing_or_partial_pricing(self):
        models = {
            "missing": {"input_cost": None, "output_cost": None},
            "partial": {"input_cost": 0.0},
            "invalid": {"input_cost": "unknown", "output_cost": 0.0},
        }

        assert filter_free_models(models) == {}


# ==============================================================================
# ModelMappingLoader.save() / update_model_mapping()
# ==============================================================================


class TestModelMappingLoaderSave:
    def test_save_creates_file_if_missing(self, tmp_path):
        path = tmp_path / "models.yaml"
        loader = ModelMappingLoader(str(path))
        mapping = {
            "display_name": "minimax-m3",
            "description": "test",
            "providers": {"openrouter": "minimax/minimax-m3"},
        }
        loader.save("minimax-m3", mapping)
        assert path.exists()
        content = path.read_text()
        assert "minimax-m3:" in content
        assert "minimax/minimax-m3" in content

    def test_save_updates_existing_entry(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text(
            "models:\n  glm-5:\n    display_name: old\n    providers:\n      openrouter: z-ai/glm-5\n"
        )
        loader = ModelMappingLoader(str(path))
        loader.save(
            "glm-5",
            {
                "display_name": "new",
                "description": "new desc",
                "providers": {"openrouter": "z-ai/glm-5-new"},
            },
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["models"]["glm-5"]["display_name"] == "new"
        assert data["models"]["glm-5"]["description"] == "new desc"
        assert data["models"]["glm-5"]["providers"]["openrouter"] == "z-ai/glm-5-new"

    def test_save_inserts_new_model_into_existing_file(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text(
            "models:\n  glm-5:\n    display_name: zai-glm-5\n    providers:\n      openrouter: z-ai/glm-5\n"
        )
        loader = ModelMappingLoader(str(path))
        loader.save(
            "claude-opus-4-8",
            {
                "display_name": "claude-opus-4-8",
                "description": "Claude Opus 4.8",
                "providers": {"openrouter": "anthropic/claude-opus-4.8"},
            },
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        # Both entries should be present
        assert "glm-5" in data["models"]
        assert "claude-opus-4-8" in data["models"]
        assert (
            data["models"]["claude-opus-4-8"]["providers"]["openrouter"]
            == "anthropic/claude-opus-4.8"
        )

    def test_save_dry_run_does_not_write(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text("models:\n  glm-5: foo\n")
        original = path.read_text()
        loader = ModelMappingLoader(str(path))
        loader.save("glm-5", {"display_name": "new", "providers": {}}, dry_run=True)
        assert path.read_text() == original
        # No backup either
        assert not path.with_suffix(".yaml.backup").exists()

    def test_save_creates_backup(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text("models:\n  glm-5: foo\n")
        loader = ModelMappingLoader(str(path))
        loader.save("glm-5", {"display_name": "new", "providers": {}})
        backup = path.with_suffix(".yaml.backup")
        assert backup.exists()
        assert "glm-5: foo" in backup.read_text()

    def test_update_model_mapping_in_memory_only(self, tmp_path):
        path = tmp_path / "models.yaml"
        path.write_text("models:\n  glm-5: foo\n")
        loader = ModelMappingLoader(str(path))
        loader.update_model_mapping("glm-5", {"display_name": "x", "providers": {}})
        # File should not be touched
        assert path.read_text() == "models:\n  glm-5: foo\n"
        # In-memory cache should be updated
        assert loader.get_model_mapping("glm-5") == {
            "display_name": "x",
            "providers": {},
        }

    def test_save_writes_valid_yaml(self, tmp_path):
        path = tmp_path / "models.yaml"
        loader = ModelMappingLoader(str(path))
        loader.save(
            "minimax-m3",
            {
                "display_name": "minimax-m3",
                "description": "Minimax M3",
                "providers": {
                    "openrouter": "minimax/minimax-m3",
                    "kilo": "minimax/minimax-m3",
                },
            },
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert (
            data["models"]["minimax-m3"]["providers"]["openrouter"]
            == "minimax/minimax-m3"
        )
        assert data["models"]["minimax-m3"]["providers"]["kilo"] == "minimax/minimax-m3"


# ==============================================================================
# ModelsPopulator integration
# ==============================================================================


class TestModelsPopulator:
    def _make_providers_yaml(self, tmp_path):
        providers = {
            "providers": {
                "alpha": {
                    "name": "Alpha",
                    "api_url": "https://alpha.test/v1/models",
                    "model_prefix": "alpha/",
                    "model_detection": {"type": "prefix", "value": "alpha/"},
                    "pricing": {
                        "input_field": "pricing.prompt",
                        "output_field": "pricing.completion",
                        "is_per_million": False,
                        "free_model_handling": True,
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                    "api_base_config": None,
                    "api_key_env": None,
                },
                "beta": {
                    "name": "Beta",
                    "api_url": "https://beta.test/v1/models",
                    "model_prefix": "beta/",
                    "model_detection": {"type": "prefix", "value": "beta/"},
                    "pricing": {
                        "input_field": None,
                        "output_field": None,
                        "is_per_million": False,
                        "free_model_handling": True,
                        "default_cost": 1.0e-09,
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                    "api_base_config": None,
                    "api_key_env": None,
                },
            }
        }
        providers_path = tmp_path / "providers.yaml"
        with open(providers_path, "w") as f:
            yaml.dump(providers, f)
        return providers_path

    def test_populate_new_model(self, tmp_path, monkeypatch):
        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        from cleanup_base import ConfigDrivenModelCleaner

        def fake_fetch(self):
            key = self.PROVIDER_NAME.lower()
            return {
                "alpha": {"alpha/test-model": {"id": "alpha/test-model"}},
                "beta": {"beta/test-model": {"id": "beta/test-model"}},
            }.get(key, {})

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
            verbose=False,
        )

        result = populator.populate("test-model", display_name="Test Model")
        providers = result["providers"]
        assert providers["alpha"] == "alpha/test-model"
        assert providers["beta"] == "beta/test-model"

    def test_free_models_only_provider_filters_before_matching(
        self, tmp_path, monkeypatch
    ):
        providers_path = self._make_providers_yaml(tmp_path)
        providers = yaml.safe_load(providers_path.read_text())
        providers["providers"]["alpha"]["free_models_only"] = True
        providers_path.write_text(yaml.dump(providers))

        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        from cleanup_base import ConfigDrivenModelCleaner

        def fake_fetch(self):
            return {
                "alpha/free-model": {
                    "id": "alpha/free-model",
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                },
                "alpha/paid-model": {
                    "id": "alpha/paid-model",
                    "input_cost": 0.1,
                    "output_cost": 0.2,
                },
            }

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
        )
        result = populator.populate("paid-model")
        assert "alpha" not in result["providers"]

    def test_populate_dry_run_does_not_write(self, tmp_path, monkeypatch):
        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        original = "models:\n  other: foo\n"
        models_path.write_text(original)

        from cleanup_base import ConfigDrivenModelCleaner

        def fake_fetch(self):
            key = self.PROVIDER_NAME.lower()
            return {"alpha": {"alpha/x": {"id": "alpha/x"}}, "beta": {}}.get(key, {})

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
        )
        populator.populate("x")
        assert models_path.read_text() == original

    def test_populate_with_provider_filter(self, tmp_path, monkeypatch):
        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        from cleanup_base import ConfigDrivenModelCleaner

        fetched = []

        def fake_fetch(self):
            key = self.PROVIDER_NAME.lower()
            fetched.append(key)
            return {
                "alpha": {"alpha/x": {"id": "alpha/x"}},
                "beta": {"beta/x": {"id": "beta/x"}},
            }.get(key, {})

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
        )
        result = populator.populate("x", provider_filter=["alpha"])
        assert "alpha" in result["providers"]
        assert "beta" not in result["providers"]
        assert "alpha" in fetched
        assert "beta" not in fetched

    def test_populate_writes_new_model_to_file(self, tmp_path, monkeypatch):
        """End-to-end: populate should actually write the new entry to disk."""
        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        models_path.write_text(
            "models:\n  other-model:\n    providers:\n      alpha: a/other\n"
        )

        from cleanup_base import ConfigDrivenModelCleaner

        def fake_fetch(self):
            key = self.PROVIDER_NAME.lower()
            return {
                "alpha": {"alpha/brand-new": {"id": "alpha/brand-new"}},
                "beta": {},
            }.get(key, {})

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=False,
        )
        populator.populate("brand-new")

        with open(models_path) as f:
            data = yaml.safe_load(f)
        assert "other-model" in data["models"]
        assert "brand-new" in data["models"]
        assert data["models"]["brand-new"]["providers"]["alpha"] == "alpha/brand-new"
        # beta is missing — should be omitted from the file entirely,
        # not written as null.
        assert "beta" not in data["models"]["brand-new"]["providers"]


# ==============================================================================
# OpenCode Go API prefix tests
# ==============================================================================


class TestOpenCodeGoApiPrefix:
    """Tests that opencode-go models get the correct openai/ or anthropic/ prefix."""

    def _make_providers_yaml(self, tmp_path) -> Path:
        """Create a minimal providers.yaml with opencode-go."""
        providers = {
            "providers": {
                "opencode-go": {
                    "name": "OpenCode Go",
                    "api_url": "https://opencode.ai/zen/go/v1/models",
                    "model_prefix": "openai/",
                    "model_detection": {"type": "api_base", "value": "opencode.ai/zen/go"},
                    "model_prefixes": [
                        {"prefix": "openai/", "api_base": "https://opencode.ai/zen/go/v1"},
                        {"prefix": "anthropic/", "api_base": "https://opencode.ai/zen/go"},
                        {
                            "prefix": "text-completion-openai/",
                            "api_base": "https://opencode.ai/zen/go/v1/responses",
                        },
                    ],
                    "responses_model_ids": ["gpt-5.6-luna"],
                    "pricing": {
                        "input_field": None,
                        "output_field": None,
                        "is_per_million": False,
                        "free_model_handling": True,
                        "default_cost": None,
                        "models_dev_id": "opencode-go",
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                    "api_base_config": {"url": "https://opencode.ai/zen/go/v1", "api_key_env": "OPENCODE_API_KEY"},
                    "api_key_env": "OPENCODE_API_KEY",
                },
            }
        }
        providers_path = tmp_path / "providers.yaml"
        with open(providers_path, "w") as f:
            yaml.dump(providers, f)
        return providers_path

    def test_opencode_go_anthropic_prefix(self, tmp_path, monkeypatch):
        """Models with provider.npm=@ai-sdk/anthropic get anthropic/ prefix."""
        from cleanup_base import ConfigDrivenModelCleaner, _models_dev_client

        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        # Mock fetch_available_models to return a model
        def fake_fetch(self):
            return {"minimax-m3": {"id": "minimax-m3"}}

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        # Mock models.dev to return anthropic provider npm
        monkeypatch.setattr(
            _models_dev_client,
            "get_model_provider_npm",
            lambda provider_id, model_id, logger=None: "@ai-sdk/anthropic",
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
            verbose=False,
        )
        result = populator.populate("minimax-m3")
        assert result["providers"]["opencode-go"] == "anthropic/minimax-m3"

    def test_opencode_go_openai_prefix(self, tmp_path, monkeypatch):
        """Models with provider.npm absent or @ai-sdk/openai get openai/ prefix."""
        from cleanup_base import ConfigDrivenModelCleaner, _models_dev_client

        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        def fake_fetch(self):
            return {"kimi-k3": {"id": "kimi-k3"}}

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        # Mock models.dev to return None (no provider field)
        monkeypatch.setattr(
            _models_dev_client,
            "get_model_provider_npm",
            lambda provider_id, model_id, logger=None: None,
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
            verbose=False,
        )
        result = populator.populate("kimi-k3")
        assert result["providers"]["opencode-go"] == "openai/kimi-k3"

    def test_opencode_go_grok_openai_prefix(self, tmp_path, monkeypatch):
        """Models with provider.npm=@ai-sdk/openai get openai/ prefix."""
        from cleanup_base import ConfigDrivenModelCleaner, _models_dev_client

        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        def fake_fetch(self):
            return {"grok-4.5": {"id": "grok-4.5"}}

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        # Mock models.dev to return @ai-sdk/openai
        monkeypatch.setattr(
            _models_dev_client,
            "get_model_provider_npm",
            lambda provider_id, model_id, logger=None: "@ai-sdk/openai",
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
            verbose=False,
        )
        result = populator.populate("grok-4.5")
        assert result["providers"]["opencode-go"] == "openai/grok-4.5"

    def test_opencode_go_responses_prefix(self, tmp_path, monkeypatch):
        """Responses models use the dedicated LiteLLM route prefix."""
        from cleanup_base import ConfigDrivenModelCleaner, _models_dev_client

        providers_path = self._make_providers_yaml(tmp_path)
        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        monkeypatch.setattr(
            ConfigDrivenModelCleaner,
            "fetch_available_models",
            lambda self: {"gpt-5.6-luna": {"id": "gpt-5.6-luna"}},
        )
        monkeypatch.setattr(
            _models_dev_client,
            "get_model_provider_npm",
            lambda provider_id, model_id, logger=None: "@ai-sdk/openai",
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
        )
        result = populator.populate("gpt-5.6-luna")

        assert result["providers"]["opencode-go"] == (
            "text-completion-openai/gpt-5.6-luna"
        )

    def test_non_opencode_go_unaffected(self, tmp_path, monkeypatch):
        """Non-opencode-go providers are not affected by the prefix logic."""
        providers = {
            "providers": {
                "openrouter": {
                    "name": "OpenRouter",
                    "api_url": "https://openrouter.ai/api/v1/models",
                    "model_prefix": "openrouter/",
                    "model_detection": {"type": "prefix", "value": "openrouter/"},
                    "pricing": {
                        "input_field": "pricing.prompt",
                        "output_field": "pricing.completion",
                        "is_per_million": False,
                        "free_model_handling": True,
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                    "api_base_config": None,
                    "api_key_env": None,
                },
            }
        }
        providers_path = tmp_path / "providers.yaml"
        with open(providers_path, "w") as f:
            yaml.dump(providers, f)

        models_path = tmp_path / "models.yaml"
        models_path.write_text("models:\n")

        from cleanup_base import ConfigDrivenModelCleaner

        def fake_fetch(self):
            return {"minimax/minimax-m3": {"id": "minimax/minimax-m3"}}

        monkeypatch.setattr(
            ConfigDrivenModelCleaner, "fetch_available_models", fake_fetch
        )

        populator = ModelsPopulator(
            providers_config_path=str(providers_path),
            models_config_path=str(models_path),
            dry_run=True,
            verbose=False,
        )
        result = populator.populate("minimax-m3")
        # openrouter should NOT get an openai/ or anthropic/ prefix
        assert result["providers"]["openrouter"] == "minimax/minimax-m3"


# ==============================================================================
# ModelsDevClient.get_model_provider_npm tests
# ==============================================================================


class TestGetModelProviderNpm:
    """Tests for ModelsDevClient.get_model_provider_npm()."""

    def test_returns_anthropic_npm(self):
        from cleanup_base import ModelsDevClient

        client = ModelsDevClient()
        client._data = {
            "opencode-go": {
                "models": {
                    "minimax-m3": {
                        "provider": {"npm": "@ai-sdk/anthropic"},
                    }
                }
            }
        }
        result = client.get_model_provider_npm("opencode-go", "minimax-m3")
        assert result == "@ai-sdk/anthropic"

    def test_returns_openai_npm(self):
        from cleanup_base import ModelsDevClient

        client = ModelsDevClient()
        client._data = {
            "opencode-go": {
                "models": {
                    "grok-4.5": {
                        "provider": {"npm": "@ai-sdk/openai"},
                    }
                }
            }
        }
        result = client.get_model_provider_npm("opencode-go", "grok-4.5")
        assert result == "@ai-sdk/openai"

    def test_returns_none_when_no_provider_field(self):
        from cleanup_base import ModelsDevClient

        client = ModelsDevClient()
        client._data = {
            "opencode-go": {
                "models": {
                    "kimi-k3": {
                        "id": "kimi-k3",
                        # No "provider" field
                    }
                }
            }
        }
        result = client.get_model_provider_npm("opencode-go", "kimi-k3")
        assert result is None

    def test_returns_none_when_model_not_found(self):
        from cleanup_base import ModelsDevClient

        client = ModelsDevClient()
        client._data = {
            "opencode-go": {
                "models": {}
            }
        }
        result = client.get_model_provider_npm("opencode-go", "nonexistent")
        assert result is None

    def test_returns_none_when_data_not_loaded(self):
        from cleanup_base import ModelsDevClient

        client = ModelsDevClient()
        # Prevent _ensure_loaded from fetching real API data
        client._load_failed = True
        result = client.get_model_provider_npm("opencode-go", "minimax-m3")
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
