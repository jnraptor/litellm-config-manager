#!/usr/bin/env python3
"""
Tests for the 4 key uncovered areas identified in coverage analysis:
1. cleanup_models.py unified script
2. File I/O and save operations
3. Free variant handling
4. ModelMappingLoader
5. Provider deletion (--delete-provider)
"""

import json
import logging
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup_base import (
    BaseModelCleaner,
    ConfigDrivenModelCleaner,
    ModelMappingLoader,
    ProviderConfigLoader,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test to ensure clean state."""
    # Reset ProviderConfigLoader singleton
    ProviderConfigLoader._instance = None
    ProviderConfigLoader._config = {}
    ProviderConfigLoader._config_path = None

    yield

    # Reset again after test
    ProviderConfigLoader._instance = None
    ProviderConfigLoader._config = {}
    ProviderConfigLoader._config_path = None


# ==============================================================================
# 1. cleanup_models.py unified script tests
# ==============================================================================


class TestCleanupModelsUnifiedScript:
    """Tests for the cleanup_models.py UnifiedModelCleaner class."""

    @pytest.fixture
    def temp_config_with_models(self, tmp_path):
        """Create temporary config with models from actual providers."""
        config_content = {
            "general_settings": {"store_prompts_in_spend_logs": True},
            "model_list": [
                {
                    "model_name": "or-model1",
                    "litellm_params": {
                        "model": "openrouter/anthropic/claude-3",
                        "order": 5,
                        "input_cost_per_token": 1e-06,
                        "output_cost_per_token": 2e-06,
                    },
                },
                {
                    "model_name": "kilo-model2",
                    "litellm_params": {
                        "model": "openai/test-model",
                        "api_base": "https://api.kilo.ai/api/gateway",
                        "api_key": "os.environ/KILO_API_KEY",
                        "order": 2,
                    },
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        return str(config_file)

    def test_unified_cleaner_loads_multiple_providers(self, temp_config_with_models):
        """Test that UnifiedModelCleaner can load multiple providers."""
        from cleanup_models import UnifiedModelCleaner

        cleaner = UnifiedModelCleaner(
            config_path=temp_config_with_models,
            provider_names=["openrouter", "kilo"],
            dry_run=True,
        )

        assert len(cleaner.cleaners) == 2
        assert "openrouter" in cleaner.cleaners
        assert "kilo" in cleaner.cleaners

    def test_unified_cleaner_extracts_provider_models(self, temp_config_with_models):
        """Test that UnifiedModelCleaner extracts models from all providers."""
        from cleanup_models import UnifiedModelCleaner

        cleaner = UnifiedModelCleaner(
            config_path=temp_config_with_models,
            provider_names=["openrouter", "kilo"],
            dry_run=True,
        )

        config = cleaner.load_config()

        # Each cleaner should have extract_provider_models method
        assert hasattr(cleaner.cleaners["openrouter"], "extract_provider_models")
        assert hasattr(cleaner.cleaners["kilo"], "extract_provider_models")

        # Extract models from config
        or_models = cleaner.cleaners["openrouter"].extract_provider_models(config)
        kilo_models = cleaner.cleaners["kilo"].extract_provider_models(config)

        assert len(or_models) == 1
        assert len(kilo_models) == 1

    def test_unified_cleaner_single_provider_mode(self, temp_config_with_models):
        """Test UnifiedModelCleaner with single provider."""
        from cleanup_models import UnifiedModelCleaner

        cleaner = UnifiedModelCleaner(
            config_path=temp_config_with_models,
            provider_names=["openrouter"],
            dry_run=True,
        )

        assert len(cleaner.cleaners) == 1
        assert "openrouter" in cleaner.cleaners

    def test_delete_model_from_config_removes_entries(self, tmp_path):
        """Test delete_model_from_config removes matching entries by model_name."""
        from cleanup_models import UnifiedModelCleaner

        config_content = {
            "model_list": [
                {
                    "model_name": "model-a",
                    "litellm_params": {"model": "openai/model-a"},
                },
                {
                    "model_name": "model-b",
                    "litellm_params": {"model": "openai/model-b"},
                },
                {
                    "model_name": "model-a",
                    "litellm_params": {"model": "openai/model-a-v2"},
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=True,
        )
        config = cleaner.load_config()
        updated_config, removed = cleaner.delete_model_from_config(config, ["model-a"])

        assert removed == 2
        remaining = [e["model_name"] for e in updated_config["model_list"]]
        assert "model-a" not in remaining
        assert len(updated_config["model_list"]) == 1
        assert remaining == ["model-b"]

    def test_delete_nonexistent_model_warns(self, tmp_path, caplog):
        """Test delete_model_from_config warns when model not found."""
        from cleanup_models import UnifiedModelCleaner

        config_content = {
            "model_list": [
                {
                    "model_name": "model-a",
                    "litellm_params": {"model": "openai/model-a"},
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=True,
        )
        config = cleaner.load_config()

        with caplog.at_level(logging.WARNING):
            _, removed = cleaner.delete_model_from_config(config, ["nonexistent"])

        assert removed == 0
        assert "not found in configuration" in caplog.text

    def test_delete_multiple_models(self, tmp_path):
        """Test delete_model_from_config removes multiple model names at once."""
        from cleanup_models import UnifiedModelCleaner

        config_content = {
            "model_list": [
                {
                    "model_name": "model-a",
                    "litellm_params": {"model": "openai/model-a"},
                },
                {
                    "model_name": "model-b",
                    "litellm_params": {"model": "openai/model-b"},
                },
                {
                    "model_name": "model-c",
                    "litellm_params": {"model": "openai/model-c"},
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=True,
        )
        config = cleaner.load_config()
        _, removed = cleaner.delete_model_from_config(config, ["model-a", "model-c"])

        assert removed == 2
        remaining = [e["model_name"] for e in config["model_list"]]
        assert remaining == ["model-b"]

    @pytest.fixture
    def temp_providers_config(self, tmp_path):
        """Create a temporary providers.yaml for delete-provider tests."""
        providers_content = {
            "providers": {
                "openrouter": {
                    "name": "OpenRouter",
                    "api_url": "https://api.test.com/models",
                    "model_prefix": "openrouter/",
                    "model_detection": {"type": "prefix", "value": "openrouter/"},
                    "pricing": {
                        "input_field": None,
                        "output_field": None,
                        "is_per_million": False,
                        "free_model_handling": True,
                        "default_cost": None,
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                    "api_base_config": None,
                    "api_key_env": None,
                    "order": 5,
                    "enabled": True,
                },
                "kilo": {
                    "name": "Kilo",
                    "api_url": "https://api.test.com/models",
                    "model_prefix": "openai/",
                    "model_detection": {
                        "type": "api_base",
                        "value": "api.kilo.ai",
                    },
                    "pricing": {
                        "input_field": None,
                        "output_field": None,
                        "is_per_million": False,
                        "free_model_handling": True,
                        "default_cost": None,
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                    "api_base_config": {
                        "url": "https://api.kilo.ai/api/gateway",
                        "api_key_env": "KILO_API_KEY",
                    },
                    "api_key_env": "KILO_API_KEY",
                    "order": 5,
                    "enabled": True,
                },
            },
            "defaults": {
                "timeout": 30,
                "retry_count": 3,
                "log_level": "INFO",
                "cost_comparison_precision": 2,
                "free_model_cost": 1.0e-09,
                "free_order": 3,
            },
        }
        providers_file = tmp_path / "providers.yaml"
        with open(providers_file, "w") as f:
            yaml.dump(providers_content, f)
        # Initialize the singleton with the temp path so all cleaners use it.
        ProviderConfigLoader(str(providers_file))
        return str(providers_file)

    def test_delete_provider_from_config_removes_prefix_models(
        self, tmp_path, temp_providers_config
    ):
        """Test delete_provider_from_config removes prefix-detected models."""
        from cleanup_models import UnifiedModelCleaner

        config_content = {
            "model_list": [
                {
                    "model_name": "or-model1",
                    "litellm_params": {"model": "openrouter/anthropic/claude-3"},
                },
                {
                    "model_name": "kilo-model1",
                    "litellm_params": {
                        "model": "openai/test-model",
                        "api_base": "https://api.kilo.ai/api/gateway",
                    },
                },
                {
                    "model_name": "other-model",
                    "litellm_params": {"model": "mistral/some-model"},
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=True,
        )
        config = cleaner.load_config()
        updated_config, removed, details = cleaner.delete_provider_from_config(
            config, ["openrouter"]
        )

        assert removed == 1
        assert details["openrouter"] == 1
        assert len(updated_config["model_list"]) == 2
        remaining_models = [
            e["litellm_params"]["model"] for e in updated_config["model_list"]
        ]
        assert "openrouter/anthropic/claude-3" not in remaining_models

    def test_delete_provider_from_config_removes_api_base_models(
        self, tmp_path, temp_providers_config
    ):
        """Test delete_provider_from_config removes api_base-detected models."""
        from cleanup_models import UnifiedModelCleaner

        config_content = {
            "model_list": [
                {
                    "model_name": "or-model1",
                    "litellm_params": {"model": "openrouter/anthropic/claude-3"},
                },
                {
                    "model_name": "kilo-model1",
                    "litellm_params": {
                        "model": "openai/test-model",
                        "api_base": "https://api.kilo.ai/api/gateway",
                    },
                },
                {
                    "model_name": "kilo-model2",
                    "litellm_params": {
                        "model": "openai/another-model",
                        "api_base": "https://api.kilo.ai/api/gateway",
                    },
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["kilo"],
            dry_run=True,
        )
        config = cleaner.load_config()
        updated_config, removed, details = cleaner.delete_provider_from_config(
            config, ["kilo"]
        )

        assert removed == 2
        assert details["kilo"] == 2
        assert len(updated_config["model_list"]) == 1
        assert updated_config["model_list"][0]["model_name"] == "or-model1"

    def test_delete_provider_from_config_removes_multiple_providers(
        self, tmp_path, temp_providers_config
    ):
        """Test delete_provider_from_config handles multiple providers at once."""
        from cleanup_models import UnifiedModelCleaner

        config_content = {
            "model_list": [
                {
                    "model_name": "or-model1",
                    "litellm_params": {"model": "openrouter/anthropic/claude-3"},
                },
                {
                    "model_name": "kilo-model1",
                    "litellm_params": {
                        "model": "openai/test-model",
                        "api_base": "https://api.kilo.ai/api/gateway",
                    },
                },
                {
                    "model_name": "other-model",
                    "litellm_params": {"model": "mistral/some-model"},
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=True,
        )
        config = cleaner.load_config()
        updated_config, removed, details = cleaner.delete_provider_from_config(
            config, ["openrouter", "kilo"]
        )

        assert removed == 2
        assert details["openrouter"] == 1
        assert details["kilo"] == 1
        assert len(updated_config["model_list"]) == 1
        assert updated_config["model_list"][0]["model_name"] == "other-model"

    def test_disable_providers_sets_enabled_false(
        self, tmp_path, temp_providers_config
    ):
        """Test disable_providers writes enabled: false to providers.yaml."""
        from cleanup_models import UnifiedModelCleaner

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"model_list": []}, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=False,
        )
        disabled = cleaner.disable_providers(["openrouter"])

        assert disabled == ["openrouter"]

        with open(temp_providers_config, "r") as f:
            saved = yaml.safe_load(f)
        assert saved["providers"]["openrouter"]["enabled"] is False
        assert saved["providers"]["kilo"]["enabled"] is True

    def test_disable_providers_already_disabled(self, tmp_path, temp_providers_config):
        """Test disable_providers skips providers already disabled."""
        from cleanup_models import UnifiedModelCleaner

        # Pre-disable openrouter in the temp config
        with open(temp_providers_config, "r") as f:
            providers = yaml.safe_load(f)
        providers["providers"]["openrouter"]["enabled"] = False
        with open(temp_providers_config, "w") as f:
            yaml.dump(providers, f)

        # Re-initialize the singleton so it sees the updated file
        ProviderConfigLoader.reset()
        ProviderConfigLoader(temp_providers_config)

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"model_list": []}, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=False,
        )
        disabled = cleaner.disable_providers(["openrouter"])

        assert disabled == []

    def test_delete_provider_dry_run_preserves_files(
        self, tmp_path, temp_providers_config
    ):
        """Test delete-provider in dry-run mode leaves files unchanged."""
        from cleanup_models import UnifiedModelCleaner

        config_content = {
            "model_list": [
                {
                    "model_name": "or-model1",
                    "litellm_params": {"model": "openrouter/anthropic/claude-3"},
                },
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["openrouter"],
            dry_run=True,
        )
        config = cleaner.load_config()
        updated_config, removed, details = cleaner.delete_provider_from_config(
            config, ["openrouter"]
        )
        disabled = cleaner.disable_providers(["openrouter"])
        cleaner.save_config(updated_config)

        assert removed == 1
        assert disabled == ["openrouter"]

        with open(config_file, "r") as f:
            saved_config = yaml.safe_load(f)
        assert len(saved_config["model_list"]) == 1

        with open(temp_providers_config, "r") as f:
            saved_providers = yaml.safe_load(f)
        assert saved_providers["providers"]["openrouter"]["enabled"] is True

    def test_delete_provider_cli_errors_for_unknown_provider(
        self, tmp_path, temp_providers_config
    ):
        """Test CLI rejects unknown providers for --delete-provider."""
        from cleanup_models import main

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"model_list": []}, f)

        with patch(
            "sys.argv",
            [
                "cleanup_models.py",
                "--provider",
                "openrouter",
                "--config",
                str(config_file),
                "--delete-provider",
                "unknown-provider",
            ],
        ):
            result = main()

        assert result == 1


# ==============================================================================
# 2. File I/O and save operations tests
# ==============================================================================


class TestFileIOSaveOperations:
    """Tests for file I/O operations including save_config."""

    @pytest.fixture
    def mock_cleaner(self):
        """Create a mock cleaner for testing."""

        class MockCleaner(BaseModelCleaner):
            PROVIDER_NAME = "TestProvider"
            SPECIAL_MODELS = set()

            def extract_provider_models(self, config):
                return []

            def fetch_available_models(self):
                return {}

            def get_api_model_id(self, model_id):
                return model_id

            def create_model_entry(self, model_id, api_model_info, model_name):
                return {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"model_list": []}, f)
            config_path = f.name

        cleaner = MockCleaner(config_path=config_path, verbose=True)
        return cleaner

    def test_save_config_creates_file(self, mock_cleaner, tmp_path):
        """Test that save_config creates the output file."""
        config = {
            "general_settings": {"test": True},
            "model_list": [
                {"model_name": "test-model", "litellm_params": {"model": "test/model"}}
            ],
        }

        output_path = tmp_path / "output.yaml"
        mock_cleaner.config_path = str(output_path)

        mock_cleaner.save_config(config)

        assert output_path.exists()

        # Verify content
        with open(output_path, "r") as f:
            saved = yaml.safe_load(f)
        assert saved["general_settings"]["test"] is True

    def test_save_config_preserves_yaml_structure(self, mock_cleaner, tmp_path):
        """Test that save_config preserves YAML structure."""
        config = {
            "model_list": [
                {
                    "model_name": "test-model",
                    "litellm_params": {
                        "model": "test/model",
                        "input_cost_per_token": 1e-06,
                        "output_cost_per_token": 2e-06,
                    },
                }
            ]
        }

        output_path = tmp_path / "output.yaml"
        mock_cleaner.config_path = str(output_path)

        mock_cleaner.save_config(config)

        with open(output_path, "r") as f:
            content = f.read()

        # Check that scientific notation is preserved
        assert "1e-06" in content or "1.0e-06" in content

    def test_load_config_parses_yaml(self, mock_cleaner):
        """Test that load_config correctly parses YAML."""
        # Write test config
        test_config = {
            "general_settings": {"store_prompts_in_spend_logs": True},
            "model_list": [
                {"model_name": "test", "litellm_params": {"model": "test/model"}}
            ],
        }
        with open(mock_cleaner.config_path, "w") as f:
            yaml.dump(test_config, f)

        loaded = mock_cleaner.load_config()

        assert loaded["general_settings"]["store_prompts_in_spend_logs"] is True
        assert len(loaded["model_list"]) == 1

    def test_preview_add_model_outputs_to_console(self, mock_cleaner, caplog):
        """Test that preview_add_model logs output."""
        caplog.set_level(logging.INFO)

        api_models = {
            "new-model": {"id": "new-model", "input_cost": 1e-06, "output_cost": 2e-06}
        }

        mock_cleaner.preview_add_model(["new-model"], api_models)

        assert (
            "Would add" in caplog.text
            or "Preview" in caplog.text
            or "new-model" in caplog.text
        )


# ==============================================================================
# 3. Free variant handling tests
# ==============================================================================


class TestFreeVariantHandling:
    """Tests for free variant handling (e.g., :free suffix for OpenRouter/Kilo)."""

    def test_check_for_free_variant_found(self):
        """Test detecting free variant when it exists."""
        # Create a cleaner that uses actual kilo provider (has free_variant_suffix)
        cleaner = ConfigDrivenModelCleaner("kilo", "config.yaml", dry_run=True)

        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
            "model1:free": {"id": "model1:free", "input_cost": 0.0, "output_cost": 0.0},
        }

        free_variant = cleaner.check_for_free_variant("model1", api_models)

        assert free_variant == "model1:free"

    def test_check_for_free_variant_not_found(self):
        """Test when free variant doesn't exist."""
        cleaner = ConfigDrivenModelCleaner("kilo", "config.yaml", dry_run=True)

        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
        }

        free_variant = cleaner.check_for_free_variant("model1", api_models)

        assert free_variant is None

    def test_check_for_free_variant_no_suffix_configured(self):
        """Test that None is returned when no suffix configured."""
        # Use nvidia provider which doesn't have free_variant_suffix
        cleaner = ConfigDrivenModelCleaner("nvidia", "config.yaml", dry_run=True)

        api_models = {"model1:free": {"id": "model1:free"}}
        free_variant = cleaner.check_for_free_variant("model1", api_models)

        assert free_variant is None

    def test_add_model_with_free_variant(self):
        """Test adding a model automatically adds its free variant."""
        cleaner = ConfigDrivenModelCleaner("kilo", "config.yaml", dry_run=True)

        config = {"model_list": []}
        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
            "model1:free": {"id": "model1:free", "input_cost": 0.0, "output_cost": 0.0},
        }

        updated_config, added = cleaner.add_model_to_config(
            config, ["model1"], api_models
        )

        # Should add both the regular model and the free variant
        assert len(added) == 2
        assert "model1" in added
        assert "model1:free" in added
        assert len(updated_config["model_list"]) == 2

    def test_add_model_free_variant_already_exists(self):
        """Test adding model when free variant already exists."""
        cleaner = ConfigDrivenModelCleaner("kilo", "config.yaml", dry_run=True)

        config = {
            "model_list": [
                {
                    "model_name": "model1",
                    "litellm_params": {"model": "openai/model1:free"},
                }
            ]
        }
        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
            "model1:free": {"id": "model1:free", "input_cost": 0.0, "output_cost": 0.0},
        }

        # Mock the extraction to show model1:free already exists
        original_extract = cleaner.extract_provider_models
        cleaner.extract_provider_models = lambda c: [
            (0, "openai/model1:free", "model1")
        ]

        updated_config, added = cleaner.add_model_to_config(
            config, ["model1"], api_models
        )

        # Restore original method
        cleaner.extract_provider_models = original_extract

        # Should only add the regular model, skip the free variant
        assert "model1" in added


# ==============================================================================
# 4. ModelMappingLoader tests
# ==============================================================================


class TestModelMappingLoader:
    """Tests for ModelMappingLoader class."""

    @pytest.fixture
    def temp_models_yaml(self, tmp_path):
        """Create a temporary models.yaml file."""
        models_content = {
            "models": {
                "glm-5": {
                    "display_name": "zai-glm-5",
                    "description": "GLM-5 model by Z.ai",
                    "providers": {
                        "openrouter": "z-ai/glm-5",
                        "kilo": "z-ai/glm-5",
                        "vercel": "zai/glm-5",
                        "poe": "glm-5",
                        "nvidia": "z-ai/glm5",
                    },
                },
                "claude-opus-4.6": {
                    "display_name": "claude-opus-4.6",
                    "description": "Claude Opus 4.6",
                    "providers": {
                        "openrouter": "anthropic/claude-opus-4.6",
                        "kilo": "anthropic/claude-opus-4.6",
                    },
                },
            }
        }
        models_file = tmp_path / "models.yaml"
        with open(models_file, "w") as f:
            yaml.dump(models_content, f)
        return str(models_file)

    def test_load_models_mapping(self, temp_models_yaml):
        """Test loading models from YAML."""
        loader = ModelMappingLoader(temp_models_yaml)

        models = loader.list_mapped_models()
        assert "glm-5" in models
        assert "claude-opus-4.6" in models

    def test_get_model_mapping(self, temp_models_yaml):
        """Test getting mapping for specific model."""
        loader = ModelMappingLoader(temp_models_yaml)

        mapping = loader.get_model_mapping("glm-5")
        assert mapping is not None
        assert mapping["display_name"] == "zai-glm-5"
        assert "openrouter" in mapping["providers"]

    def test_get_model_mapping_missing(self, temp_models_yaml):
        """Test getting mapping for non-existent model."""
        loader = ModelMappingLoader(temp_models_yaml)

        mapping = loader.get_model_mapping("nonexistent-model")
        assert mapping is None

    def test_get_provider_model_id(self, temp_models_yaml):
        """Test getting provider-specific model ID."""
        loader = ModelMappingLoader(temp_models_yaml)

        openrouter_id = loader.get_provider_model_id("glm-5", "openrouter")
        assert openrouter_id == "z-ai/glm-5"

        kilo_id = loader.get_provider_model_id("glm-5", "kilo")
        assert kilo_id == "z-ai/glm-5"

    def test_get_provider_model_id_missing(self, temp_models_yaml):
        """Test getting ID when provider not configured for model."""
        loader = ModelMappingLoader(temp_models_yaml)

        # Requesty is not in the glm-5 providers
        result = loader.get_provider_model_id("glm-5", "requesty")
        assert result is None

    def test_list_mapped_models(self, temp_models_yaml):
        """Test getting all mapped models."""
        loader = ModelMappingLoader(temp_models_yaml)

        models = loader.list_mapped_models()
        assert "glm-5" in models
        assert "claude-opus-4.6" in models
        assert len(models) == 2

    def test_display_name_normalization(self, temp_models_yaml):
        """Test that display names are properly normalized."""
        loader = ModelMappingLoader(temp_models_yaml)

        mapping = loader.get_model_mapping("glm-5")
        # Display name should be usable as model_name
        assert " " not in mapping["display_name"]  # No spaces
        assert mapping["display_name"] == "zai-glm-5"


# ==============================================================================
# Integration tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_cleanup_workflow(self, tmp_path):
        """Test a complete cleanup workflow using actual providers.yaml."""
        # Create config.yaml
        config_content = {
            "model_list": [
                {
                    "model_name": "or-valid-model",
                    "litellm_params": {
                        "model": "openrouter/anthropic/claude-3",
                        "input_cost_per_token": 1e-06,
                        "output_cost_per_token": 2e-06,
                        "order": 5,
                    },
                },
                {
                    "model_name": "or-invalid-model",
                    "litellm_params": {
                        "model": "openrouter/invalid/nonexistent-model",
                        "input_cost_per_token": 1e-06,
                        "output_cost_per_token": 2e-06,
                        "order": 5,
                    },
                },
            ]
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        class TestCleaner(ConfigDrivenModelCleaner):
            def fetch_available_models(self):
                return {
                    "anthropic/claude-3": {
                        "id": "anthropic/claude-3",
                        "input_cost": 1e-06,
                        "output_cost": 2e-06,
                    }
                }

        cleaner = TestCleaner(
            provider_name="openrouter",
            config_path=str(config_file),
        )

        # Run cleanup
        config = cleaner.load_config()
        api_models = cleaner.fetch_available_models()
        config_models = cleaner.extract_provider_models(config)
        invalid_models = cleaner.validate_models(config_models, api_models)

        assert len(invalid_models) == 1
        assert invalid_models[0][1] == "openrouter/invalid/nonexistent-model"

    def test_add_mapped_model_across_providers(self, tmp_path):
        """Test adding a mapped model across multiple providers."""
        # Create models.yaml
        models_content = {
            "models": {
                "test-model": {
                    "display_name": "test-model",
                    "providers": {
                        "openrouter": "vendor/test-model",
                    },
                }
            }
        }
        models_file = tmp_path / "models.yaml"
        with open(models_file, "w") as f:
            yaml.dump(models_content, f)

        # Load model mapping
        loader = ModelMappingLoader(str(models_file))
        model_mapping = loader.get_model_mapping("test-model")

        assert model_mapping is not None
        assert "openrouter" in model_mapping["providers"]
        assert model_mapping["providers"]["openrouter"] == "vendor/test-model"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
