#!/usr/bin/env python3
"""
Tests for opencode.json regeneration from LiteLLM config.yaml.

Covers:
- build_opencode_models(): grouping/dedupe, limit merging, modalities, skip modes, sorting
- regenerate_opencode_json(): wrapper preservation, backup, dry-run, missing/invalid files
- Integration with UnifiedModelCleaner.save_config()
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup_base import (
    ProviderConfigLoader,
    build_opencode_models,
    regenerate_opencode_json,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test to ensure clean state."""
    ProviderConfigLoader._instance = None
    ProviderConfigLoader._config = {}
    ProviderConfigLoader._config_path = None

    yield

    ProviderConfigLoader._instance = None
    ProviderConfigLoader._config = {}
    ProviderConfigLoader._config_path = None


def make_entry(name, model_id, mode=None, max_input=None, max_output=None, supports=None):
    entry = {
        "model_name": name,
        "litellm_params": {"model": model_id, "order": 5},
    }
    model_info = {}
    if mode is not None:
        model_info["mode"] = mode
    if max_input is not None:
        model_info["max_input_tokens"] = max_input
    if max_output is not None:
        model_info["max_output_tokens"] = max_output
    if supports:
        model_info.update(supports)
    if model_info:
        entry["model_info"] = model_info
    return entry


# ==============================================================================
# 1. build_opencode_models unit tests
# ==============================================================================


class TestBuildOpencodeModels:
    """Tests for build_opencode_models()."""

    def test_basic_conversion(self):
        config = {
            "model_list": [
                make_entry(
                    "glm-5",
                    "openrouter/z-ai/glm-5",
                    max_input=202752,
                    max_output=8192,
                )
            ]
        }

        models = build_opencode_models(config)

        assert models == {
            "glm-5": {
                "name": "glm-5",
                "limit": {"context": 202752, "output": 8192},
                "modalities": {"input": ["text"], "output": ["text"]},
            }
        }

    def test_dedupes_shared_model_name(self):
        """Load-balanced duplicates sharing a model_name collapse to one entry."""
        config = {
            "model_list": [
                make_entry("glm-5", "openrouter/z-ai/glm-5", max_input=100000),
                make_entry("glm-5", "kilo/z-ai/glm-5", max_input=202752, max_output=8192),
            ]
        }

        models = build_opencode_models(config)

        assert list(models) == ["glm-5"]
        # Max limits across duplicate entries
        assert models["glm-5"]["limit"] == {"context": 202752, "output": 8192}

    def test_omits_missing_limits(self):
        config = {"model_list": [make_entry("m1", "openrouter/m1")]}

        models = build_opencode_models(config)

        assert "limit" not in models["m1"]

    def test_partial_limits_omitted(self):
        """A limit with only one side is schema-invalid, so it is omitted."""
        config = {
            "model_list": [make_entry("m1", "openrouter/m1", max_output=4096)]
        }

        models = build_opencode_models(config)

        assert "limit" not in models["m1"]

    def test_default_modalities_when_absent(self):
        config = {"model_list": [make_entry("m1", "openrouter/m1")]}

        models = build_opencode_models(config)

        assert models["m1"]["modalities"] == {"input": ["text"], "output": ["text"]}

    def test_skips_embedding_rerank_image_generation(self):
        config = {
            "model_list": [
                make_entry("emb", "openrouter/bge", mode="embedding"),
                make_entry("rr", "openrouter/reranker", mode="rerank"),
                make_entry("img", "openrouter/dall-e", mode="image_generation"),
                make_entry("chat", "openrouter/gpt-5"),
                make_entry("ocr", "azure_ai/mistral-document-ai", mode="ocr"),
            ]
        }

        models = build_opencode_models(config)

        assert set(models) == {"chat", "ocr"}

    def test_mixed_group_with_chat_entry_is_kept(self):
        """A model_name whose entries include a chat model is retained."""
        config = {
            "model_list": [
                make_entry("multi", "openrouter/multi", mode="embedding"),
                make_entry("multi", "kilo/multi"),
            ]
        }

        models = build_opencode_models(config)

        assert "multi" in models

    def test_keys_sorted_alphabetically(self):
        config = {
            "model_list": [
                make_entry("zeta", "openrouter/z"),
                make_entry("alpha", "openrouter/a"),
                make_entry("mid", "openrouter/m"),
            ]
        }

        models = build_opencode_models(config)

        assert list(models) == ["alpha", "mid", "zeta"]

    def test_empty_and_missing_model_list(self):
        assert build_opencode_models({}) == {}
        assert build_opencode_models({"model_list": []}) == {}
        assert build_opencode_models({"model_list": None}) == {}

    def test_modalities_union_merged_across_duplicates(self):
        """Duplicate entries union their modalities instead of first-wins."""
        config = {
            "model_list": [
                make_entry("m", "openrouter/m"),
                make_entry(
                    "m", "kilo/m", supports={"supports_vision": True}
                ),
            ]
        }

        models = build_opencode_models(config)

        assert models["m"]["modalities"] == {
            "input": ["text", "image"],
            "output": ["text"],
        }

    def test_modalities_union_keeps_capabilities_from_both_entries(self):
        """Vision from one entry and audio from another are both preserved."""
        config = {
            "model_list": [
                make_entry("m", "openrouter/m", supports={"supports_vision": True}),
                make_entry(
                    "m",
                    "kilo/m",
                    supports={
                        "supports_audio_input": True,
                        "supports_audio_output": True,
                    },
                ),
            ]
        }

        models = build_opencode_models(config)

        assert models["m"]["modalities"] == {
            "input": ["text", "image", "audio"],
            "output": ["text", "audio"],
        }

    def test_supports_audio_flags_map_to_modalities(self):
        config = {
            "model_list": [
                make_entry(
                    "m",
                    "openrouter/m",
                    supports={
                        "supports_audio_input": True,
                        "supports_audio_output": True,
                    },
                )
            ]
        }

        models = build_opencode_models(config)

        assert models["m"]["modalities"] == {
            "input": ["text", "audio"],
            "output": ["text", "audio"],
        }


# ==============================================================================
# 2. regenerate_opencode_json unit tests
# ==============================================================================


SAMPLE_OPENCODE = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "litellm": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "LiteLLM",
            "options": {"baseURL": "https://litellm.example.com/v1"},
            "models": {
                "stale-hand-written": {"name": "Stale Hand Written"},
            },
        },
        "other-provider": {"npm": "@ai-sdk/other", "models": {"keep-me": {}}},
    },
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestRegenerateOpencodeJson:
    """Tests for regenerate_opencode_json()."""

    @pytest.fixture
    def opencode_path(self, tmp_path):
        path = tmp_path / "opencode.json"
        path.write_text(json.dumps(SAMPLE_OPENCODE, indent=2))
        return path

    @pytest.fixture
    def litellm_config(self):
        return {
            "model_list": [
                make_entry(
                    "glm-5",
                    "openrouter/z-ai/glm-5",
                    max_input=202752,
                    max_output=8192,
                )
            ]
        }

    def test_replaces_models_and_preserves_wrapper(self, opencode_path, litellm_config):
        written = regenerate_opencode_json(litellm_config, opencode_path)

        assert written is True
        data = load_json(opencode_path)
        assert data["$schema"] == "https://opencode.ai/config.json"
        litellm = data["provider"]["litellm"]
        assert litellm["npm"] == "@ai-sdk/openai-compatible"
        assert litellm["name"] == "LiteLLM"
        assert litellm["options"] == {"baseURL": "https://litellm.example.com/v1"}
        # Stale hand-written model replaced by generated content
        assert "stale-hand-written" not in litellm["models"]
        assert litellm["models"]["glm-5"]["limit"] == {
            "context": 202752,
            "output": 8192,
        }
        # Other providers untouched
        assert data["provider"]["other-provider"]["models"] == {"keep-me": {}}

    def test_backup_created(self, opencode_path, litellm_config):
        original = opencode_path.read_text()

        regenerate_opencode_json(litellm_config, opencode_path)

        backup = opencode_path.with_suffix(".json.backup")
        assert backup.exists()
        assert backup.read_text() == original

    def test_noop_when_file_missing(self, tmp_path, litellm_config):
        missing = tmp_path / "nope" / "opencode.json"

        assert regenerate_opencode_json(litellm_config, missing) is False
        assert not missing.exists()

    def test_skips_when_no_litellm_section(self, tmp_path, litellm_config):
        path = tmp_path / "opencode.json"
        path.write_text(json.dumps({"$schema": "x", "provider": {"other": {}}}))
        original = path.read_text()

        assert regenerate_opencode_json(litellm_config, path) is False
        assert path.read_text() == original

    def test_skips_on_invalid_json(self, tmp_path, litellm_config):
        path = tmp_path / "opencode.json"
        path.write_text("{not valid json")

        assert regenerate_opencode_json(litellm_config, path) is False

    def test_dry_run_does_not_write(self, opencode_path, litellm_config):
        original = opencode_path.read_text()

        would_write = regenerate_opencode_json(
            litellm_config, opencode_path, dry_run=True
        )

        assert would_write is True
        assert opencode_path.read_text() == original
        assert not opencode_path.with_suffix(".json.backup").exists()

    def test_unchanged_content_returns_false(self, opencode_path, litellm_config):
        regenerate_opencode_json(litellm_config, opencode_path)
        content_after_first = opencode_path.read_text()

        second = regenerate_opencode_json(litellm_config, opencode_path)

        assert second is False
        assert opencode_path.read_text() == content_after_first

    def test_trailing_newline_written(self, opencode_path, litellm_config):
        regenerate_opencode_json(litellm_config, opencode_path)

        assert opencode_path.read_text().endswith("}\n")


# ==============================================================================
# 3. Integration with UnifiedModelCleaner.save_config
# ==============================================================================


class TestUnifiedSaveConfigSync:
    """opencode.json is regenerated whenever the unified script saves config.yaml."""

    @pytest.fixture
    def workspace(self, tmp_path):
        """Set up a temp workspace with config.yaml + opencode.json + providers/models."""
        repo_root = Path(__file__).parent.parent
        config = {
            "model_list": [
                make_entry(
                    "or-glm-5",
                    "openrouter/z-ai/glm-5",
                    max_input=202752,
                    max_output=8192,
                ),
                make_entry("or-embed", "openrouter/bge", mode="embedding"),
            ]
        }
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config, sort_keys=False))

        opencode_path = tmp_path / "opencode.json"
        opencode_path.write_text(json.dumps(SAMPLE_OPENCODE, indent=2))

        return config_path, opencode_path

    def _make_cleaner(self, config_path, dry_run=False):
        from cleanup_models import UnifiedModelCleaner

        return UnifiedModelCleaner(
            config_path=str(config_path),
            provider_names=["openrouter"],
            dry_run=dry_run,
            verbose=False,
        )

    def test_save_regenerates_opencode_json(self, workspace):
        config_path, opencode_path = workspace
        cleaner = self._make_cleaner(config_path)
        config = cleaner.load_config()

        cleaner.save_config(config)

        data = load_json(opencode_path)
        models = data["provider"]["litellm"]["models"]
        assert "or-glm-5" in models
        assert models["or-glm-5"]["limit"] == {"context": 202752, "output": 8192}
        # embedding model excluded
        assert "or-embed" not in models
        # stale hand-written entry dropped
        assert "stale-hand-written" not in models

    def test_dry_run_leaves_opencode_json_untouched(self, workspace):
        config_path, opencode_path = workspace
        original = opencode_path.read_text()
        cleaner = self._make_cleaner(config_path, dry_run=True)
        config = cleaner.load_config()

        cleaner.save_config(config)

        assert opencode_path.read_text() == original
        assert not opencode_path.with_suffix(".json.backup").exists()

    def test_missing_opencode_json_does_not_break_save(self, workspace):
        config_path, opencode_path = workspace
        opencode_path.unlink()
        cleaner = self._make_cleaner(config_path)
        config = cleaner.load_config()

        # Should save config.yaml fine and simply skip the sync
        cleaner.save_config(config)

        assert config_path.exists()
        assert not opencode_path.exists()
