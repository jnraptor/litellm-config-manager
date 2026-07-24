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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
