#!/usr/bin/env python3
"""
Additional tests for cleanup_base.py to improve code coverage.

Tests focus on:
- Model validation and invalid model detection
- Cost updating logic
- Model addition to config
- Model name generation
- Report generation
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup_base import (
    BaseModelCleaner,
    ConfigDrivenModelCleaner,
    ProviderConfigLoader,
    costs_are_equal,
    adjust_cost_for_free_model,
    get_nested_value,
    sort_model_list,
    is_api_base_model,
)


class MockModelCleaner(BaseModelCleaner):
    """Concrete implementation of BaseModelCleaner for testing."""

    def __init__(
        self,
        config_path: str = "test_config.yaml",
        dry_run: bool = False,
        verbose: bool = False,
    ):
        super().__init__(config_path, dry_run, verbose)
        self.PROVIDER_NAME = "TestProvider"
        self.SPECIAL_MODELS = {"special-model"}
        self.defaults = {}

    def extract_provider_models(self, config):
        """Extract models from config."""
        models = []
        for i, entry in enumerate(config.get("model_list", [])):
            model_id = entry.get("litellm_params", {}).get("model", "")
            model_name = entry.get("model_name", "")
            if model_id.startswith("test/"):
                models.append((i, model_id, model_name))
        return models

    def fetch_available_models(self):
        """Fetch models from API."""
        return {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
            "model2": {"id": "model2", "input_cost": 3e-06, "output_cost": 4e-06},
        }

    def get_api_model_id(self, model_id):
        """Extract API model ID."""
        if model_id.startswith("test/"):
            return model_id[5:]
        return model_id

    def create_model_entry(self, model_id, api_model_info, model_name):
        """Create a model entry."""
        return {
            "model_name": model_name,
            "litellm_params": {
                "model": f"test/{model_id}",
                "input_cost_per_token": api_model_info.get("input_cost"),
                "output_cost_per_token": api_model_info.get("output_cost"),
            },
        }


class TestModelValidation:
    """Tests for model validation logic."""

    @pytest.fixture
    def cleaner(self):
        return MockModelCleaner(verbose=True)

    def test_validate_models_no_invalid(self, cleaner):
        """Test validation when all models are valid."""
        config_models = [
            (0, "test/model1", "model-1"),
            (1, "test/model2", "model-2"),
        ]
        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
            "model2": {"id": "model2", "input_cost": 3e-06, "output_cost": 4e-06},
        }

        invalid = cleaner.validate_models(config_models, api_models)

        assert len(invalid) == 0

    def test_validate_models_with_invalid(self, cleaner):
        """Test validation when some models are invalid."""
        config_models = [
            (0, "test/model1", "model-1"),
            (1, "test/invalid-model", "invalid"),
            (2, "test/model2", "model-2"),
        ]
        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
            "model2": {"id": "model2", "input_cost": 3e-06, "output_cost": 4e-06},
        }

        invalid = cleaner.validate_models(config_models, api_models)

        assert len(invalid) == 1
        assert invalid[0] == (1, "test/invalid-model", "invalid")

    def test_validate_models_special_models_excluded(self, cleaner):
        """Test that special models are not marked as invalid."""
        config_models = [
            (0, "test/special-model", "special"),
        ]
        api_models = {}  # Empty, but special-model should still be valid

        invalid = cleaner.validate_models(config_models, api_models)

        assert len(invalid) == 0


class TestCostValidation:
    """Tests for cost validation and updating logic."""

    @pytest.fixture
    def cleaner(self):
        return MockModelCleaner(verbose=True)

    @pytest.fixture
    def sample_config(self):
        return {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "test/model1",
                        "input_cost_per_token": 1e-06,
                        "output_cost_per_token": 2e-06,
                        "order": 1,
                    },
                },
                {
                    "model_name": "model-2",
                    "litellm_params": {
                        "model": "test/model2",
                        "input_cost_per_token": 3e-06,
                        "output_cost_per_token": 4e-06,
                        "order": 1,
                    },
                },
            ]
        }

    def test_no_cost_changes(self, cleaner, sample_config):
        """Test when costs match API."""
        config_models = [
            (0, "test/model1", "model-1"),
            (1, "test/model2", "model-2"),
        ]
        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
            "model2": {"id": "model2", "input_cost": 3e-06, "output_cost": 4e-06},
        }

        updated_config, changes, order_changed = cleaner.validate_and_update_costs(
            sample_config, config_models, api_models, provider_order=2
        )

        assert len(changes) == 0
        # Order should still be updated
        assert updated_config["model_list"][0]["litellm_params"]["order"] == 2

    def test_input_cost_changed(self, cleaner, sample_config):
        """Test when only input cost changes."""
        config_models = [
            (0, "test/model1", "model-1"),
        ]
        api_models = {
            "model1": {"id": "model1", "input_cost": 1.5e-06, "output_cost": 2e-06},
        }

        updated_config, changes, order_changed = cleaner.validate_and_update_costs(
            sample_config, config_models, api_models
        )

        assert len(changes) == 1
        assert "input_cost" in changes[0]["changes"]
        assert changes[0]["changes"]["input_cost"]["old"] == 1e-06
        assert changes[0]["changes"]["input_cost"]["new"] == 1.5e-06
        assert (
            updated_config["model_list"][0]["litellm_params"]["input_cost_per_token"]
            == 1.5e-06
        )

    def test_output_cost_changed(self, cleaner, sample_config):
        """Test when only output cost changes."""
        config_models = [
            (0, "test/model1", "model-1"),
        ]
        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2.5e-06},
        }

        updated_config, changes, order_changed = cleaner.validate_and_update_costs(
            sample_config, config_models, api_models
        )

        assert len(changes) == 1
        assert "output_cost" in changes[0]["changes"]
        assert changes[0]["changes"]["output_cost"]["old"] == 2e-06
        assert changes[0]["changes"]["output_cost"]["new"] == 2.5e-06

    def test_both_costs_changed(self, cleaner, sample_config):
        """Test when both input and output costs change."""
        config_models = [
            (0, "test/model1", "model-1"),
        ]
        api_models = {
            "model1": {"id": "model1", "input_cost": 1.5e-06, "output_cost": 2.5e-06},
        }

        updated_config, changes, order_changed = cleaner.validate_and_update_costs(
            sample_config, config_models, api_models
        )

        assert len(changes) == 1
        assert "input_cost" in changes[0]["changes"]
        assert "output_cost" in changes[0]["changes"]

    def test_new_cost_added(self, cleaner):
        """Test when model previously had no cost but now does."""
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "test/model1",
                        "order": 1,
                    },
                },
            ]
        }
        config_models = [(0, "test/model1", "model-1")]
        api_models = {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
        }

        updated_config, changes, order_changed = cleaner.validate_and_update_costs(
            config, config_models, api_models
        )

        assert len(changes) == 1
        assert "input_cost" in changes[0]["changes"]
        assert "output_cost" in changes[0]["changes"]
        assert (
            updated_config["model_list"][0]["litellm_params"]["input_cost_per_token"]
            == 1e-06
        )

    def test_api_model_not_found(self, cleaner, sample_config):
        """Test when API model is not found (should be skipped)."""
        config_models = [(0, "test/model1", "model-1")]
        api_models = {}  # Empty

        updated_config, changes, order_changed = cleaner.validate_and_update_costs(
            sample_config, config_models, api_models
        )

        assert len(changes) == 0


class TestModelAddition:
    """Tests for adding models to config."""

    @pytest.fixture
    def cleaner(self):
        return MockModelCleaner(verbose=True)

    @pytest.fixture
    def empty_config(self):
        return {"model_list": []}

    @pytest.fixture
    def api_models(self):
        return {
            "new-model": {"id": "new-model", "input_cost": 1e-06, "output_cost": 2e-06},
            "another-model": {
                "id": "another-model",
                "input_cost": 3e-06,
                "output_cost": 4e-06,
            },
        }

    def test_add_single_model(self, cleaner, empty_config, api_models):
        """Test adding a single model."""
        config, added = cleaner.add_model_to_config(
            empty_config, ["new-model"], api_models
        )

        assert len(added) == 1
        assert "new-model" in added
        assert len(config["model_list"]) == 1
        assert config["model_list"][0]["model_name"] == "new-model"

    def test_add_multiple_models(self, cleaner, empty_config, api_models):
        """Test adding multiple models."""
        config, added = cleaner.add_model_to_config(
            empty_config, ["new-model", "another-model"], api_models
        )

        assert len(added) == 2
        assert len(config["model_list"]) == 2

    def test_add_duplicate_model(self, cleaner, empty_config, api_models):
        """Test that duplicate models are not added."""
        # First add
        config, added = cleaner.add_model_to_config(
            empty_config, ["new-model"], api_models
        )

        # Try to add again
        config, added = cleaner.add_model_to_config(config, ["new-model"], api_models)

        assert len(added) == 0  # Should not add duplicate
        assert len(config["model_list"]) == 1

    def test_add_model_not_in_api(self, cleaner, empty_config, api_models):
        """Test adding a model that doesn't exist in API."""
        config, added = cleaner.add_model_to_config(
            empty_config, ["nonexistent-model"], api_models
        )

        assert len(added) == 0
        assert len(config["model_list"]) == 0

    def test_add_with_custom_name(self, cleaner, empty_config, api_models):
        """Test adding with a custom model name."""
        config, added = cleaner.add_model_to_config(
            empty_config, ["new-model"], api_models, custom_model_name="my-custom-name"
        )

        assert len(added) == 1
        assert config["model_list"][0]["model_name"] == "my-custom-name"

    def test_name_conflict_resolution(self, cleaner, api_models):
        """Test that name conflicts are resolved by appending counter."""
        config = {
            "model_list": [
                {
                    "model_name": "new-model",
                    "litellm_params": {"model": "test/existing"},
                }
            ]
        }

        # Adding a model with a different model_id but same generated name
        # The model_id "new-model" doesn't exist yet, but the name "new-model" does
        config, added = cleaner.add_model_to_config(config, ["new-model"], api_models)

        # Should add the model with a modified name (new-model-1)
        assert len(added) == 1
        assert config["model_list"][1]["model_name"] == "new-model-1"


class TestModelNameGeneration:
    """Tests for model name generation."""

    @pytest.fixture
    def cleaner(self):
        return MockModelCleaner()

    def test_generate_simple_name(self, cleaner):
        """Test generating name from simple model ID."""
        name = cleaner.generate_model_name("model-name")
        assert name == "model-name"

    def test_generate_with_slash(self, cleaner):
        """Test generating name with slash replaced."""
        name = cleaner.generate_model_name("provider/model-name")
        assert name == "provider-model-name"

    def test_generate_with_colon(self, cleaner):
        """Test generating name with colon replaced."""
        name = cleaner.generate_model_name("model:name")
        assert name == "model-name"

    def test_generate_with_prefix(self, cleaner):
        """Test generating name with prefix."""
        name = cleaner.generate_model_name("model-name", prefix="or-")
        assert name == "or-model-name"

    def test_generate_lowercase(self, cleaner):
        """Test that names are lowercased."""
        name = cleaner.generate_model_name("MODEL-NAME")
        assert name == "model-name"


class TestIsApiBaseModel:
    """Tests for is_api_base_model utility function."""

    def test_matching_api_base(self):
        """Test when api_base matches detection value."""
        result = is_api_base_model(
            api_base="https://api.kilo.ai/v1",
            model_id="openai/gpt-4",
            detection_value="api.kilo.ai",
            model_prefix="openai/",
            api_base_env_var=None,
        )
        assert result is True

    def test_non_matching_api_base(self):
        """Test when api_base doesn't match."""
        result = is_api_base_model(
            api_base="https://api.openai.com/v1",
            model_id="openai/gpt-4",
            detection_value="api.kilo.ai",
            model_prefix="openai/",
            api_base_env_var=None,
        )
        assert result is False

    def test_wrong_model_prefix(self):
        """Test when model prefix doesn't match."""
        result = is_api_base_model(
            api_base="https://api.kilo.ai/v1",
            model_id="anthropic/claude-3",
            detection_value="api.kilo.ai",
            model_prefix="openai/",
            api_base_env_var=None,
        )
        assert result is False

    def test_env_var_reference_in_config(self):
        """Test when api_base is an env var reference."""
        result = is_api_base_model(
            api_base="os.environ/KILO_API_BASE",
            model_id="openai/gpt-4",
            detection_value="api.kilo.ai",
            model_prefix="openai/",
            api_base_env_var="KILO_API_BASE",
        )
        assert result is True

    def test_env_var_mismatch(self):
        """Test when env var doesn't match."""
        result = is_api_base_model(
            api_base="os.environ/OTHER_API_BASE",
            model_id="openai/gpt-4",
            detection_value="api.kilo.ai",
            model_prefix="openai/",
            api_base_env_var="KILO_API_BASE",
        )
        assert result is False


class TestSortModelList:
    """Tests for sort_model_list utility function."""

    def test_already_sorted(self):
        """Test sorting when already sorted."""
        model_list = [
            {
                "model_name": "a-model",
                "litellm_params": {"model": "test/a", "order": 1},
            },
            {
                "model_name": "b-model",
                "litellm_params": {"model": "test/b", "order": 1},
            },
        ]

        sorted_list, was_sorted = sort_model_list(model_list)

        assert was_sorted is False
        assert sorted_list[0]["model_name"] == "a-model"

    def test_needs_sorting_by_name(self):
        """Test sorting by model name."""
        model_list = [
            {
                "model_name": "z-model",
                "litellm_params": {"model": "test/z", "order": 1},
            },
            {
                "model_name": "a-model",
                "litellm_params": {"model": "test/a", "order": 1},
            },
        ]

        sorted_list, was_sorted = sort_model_list(model_list)

        assert was_sorted is True
        assert sorted_list[0]["model_name"] == "a-model"
        assert sorted_list[1]["model_name"] == "z-model"

    def test_sort_by_order(self):
        """Test sorting by order when names are same."""
        model_list = [
            {
                "model_name": "same-name",
                "litellm_params": {"model": "test/b", "order": 5},
            },
            {
                "model_name": "same-name",
                "litellm_params": {"model": "test/a", "order": 1},
            },
        ]

        sorted_list, was_sorted = sort_model_list(model_list)

        assert was_sorted is True
        assert sorted_list[0]["litellm_params"]["order"] == 1
        assert sorted_list[1]["litellm_params"]["order"] == 5

    def test_sort_by_model_id(self):
        """Test sorting by model ID when name and order are same."""
        model_list = [
            {
                "model_name": "same-name",
                "litellm_params": {"model": "test/z", "order": 1},
            },
            {
                "model_name": "same-name",
                "litellm_params": {"model": "test/a", "order": 1},
            },
        ]

        sorted_list, was_sorted = sort_model_list(model_list)

        assert was_sorted is True
        assert sorted_list[0]["litellm_params"]["model"] == "test/a"

    def test_empty_list(self):
        """Test sorting empty list."""
        sorted_list, was_sorted = sort_model_list([])

        assert was_sorted is False
        assert sorted_list == []

    def test_case_insensitive_sort(self):
        """Test that sorting is case insensitive."""
        model_list = [
            {"model_name": "Z-model", "litellm_params": {"model": "test/z"}},
            {"model_name": "a-model", "litellm_params": {"model": "test/a"}},
        ]

        sorted_list, was_sorted = sort_model_list(model_list)

        assert was_sorted is True
        assert sorted_list[0]["model_name"] == "a-model"


class TestGetNestedValue:
    """Tests for get_nested_value utility function."""

    def test_simple_key(self):
        """Test getting simple key."""
        data = {"key": "value"}
        result = get_nested_value(data, "key")
        assert result == "value"

    def test_nested_key(self):
        """Test getting nested key."""
        data = {"pricing": {"prompt": 0.001}}
        result = get_nested_value(data, "pricing.prompt")
        assert result == 0.001

    def test_deeply_nested(self):
        """Test getting deeply nested key."""
        data = {"a": {"b": {"c": {"d": "value"}}}}
        result = get_nested_value(data, "a.b.c.d")
        assert result == "value"

    def test_missing_key(self):
        """Test when key doesn't exist."""
        data = {"key": "value"}
        result = get_nested_value(data, "missing")
        assert result is None

    def test_missing_nested_key(self):
        """Test when nested key doesn't exist."""
        data = {"pricing": {"prompt": 0.001}}
        result = get_nested_value(data, "pricing.completion")
        assert result is None

    def test_none_data(self):
        """Test when data is None (should return None)."""
        result = get_nested_value(None, "key")
        assert result is None


class TestReportGeneration:
    """Tests for report generation methods."""

    @pytest.fixture
    def cleaner(self):
        return MockModelCleaner(verbose=True)

    def test_generate_report_no_changes(self, cleaner, caplog):
        """Test report when no changes."""
        caplog.set_level(logging.INFO)

        cleaner.generate_report([], [], False)

        assert "All TestProvider models are valid" in caplog.text

    def test_generate_report_with_invalid_models(self, cleaner, caplog):
        """Test report with invalid models."""
        caplog.set_level(logging.INFO)

        invalid_models = [
            (0, "test/invalid", "invalid-model"),
        ]
        cleaner.generate_report(invalid_models, [], False)

        assert "1 invalid models removed" in caplog.text

    def test_generate_report_with_cost_changes(self, cleaner, caplog):
        """Test report with cost changes."""
        caplog.set_level(logging.INFO)

        cost_changes = [
            {
                "index": 0,
                "model_id": "test/model1",
                "model_name": "model-1",
                "changes": {
                    "input_cost": {"old": 1e-06, "new": 1.5e-06},
                },
            }
        ]
        cleaner.generate_report([], cost_changes, False)

        assert "1 models had cost changes applied" in caplog.text

    def test_generate_report_dry_run(self, cleaner, caplog):
        """Test report in dry-run mode."""
        caplog.set_level(logging.INFO)
        cleaner.dry_run = True

        invalid_models = [
            (0, "test/invalid", "invalid-model"),
        ]
        cleaner.generate_report(invalid_models, [], True)

        assert "[DRY-RUN]" in caplog.text
        assert "WOULD REMOVE" in caplog.text


class TestConfigLoadingAndSaving:
    """Tests for config loading and saving."""

    @pytest.fixture
    def temp_config_file(self, tmp_path):
        """Create a temporary config file."""
        config = {
            "general_settings": {"store_prompts_in_spend_logs": True},
            "model_list": [
                {"model_name": "test-model", "litellm_params": {"model": "test/model"}}
            ],
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            import yaml

            yaml.dump(config, f)
        return config_file

    def test_load_config_success(self, temp_config_file):
        """Test loading existing config."""
        cleaner = MockModelCleaner(config_path=str(temp_config_file))

        config = cleaner.load_config()

        assert "general_settings" in config
        assert "model_list" in config
        assert len(config["model_list"]) == 1

    def test_load_config_missing_file(self, tmp_path):
        """Test loading missing config file."""
        missing_file = tmp_path / "nonexistent.yaml"
        cleaner = MockModelCleaner(config_path=str(missing_file))

        with pytest.raises(FileNotFoundError):
            cleaner.load_config()


class TestConfigDrivenModelCleaner:
    """Tests specific to ConfigDrivenModelCleaner."""

    def test_init_loads_provider_config(self, tmp_path):
        """Test that initialization loads provider config."""
        # Create a minimal providers.yaml
        providers_content = {
            "providers": {
                "test_provider": {
                    "name": "TestProvider",
                    "description": "Test provider",
                    "api_url": "https://api.test.com/models",
                    "model_prefix": "test/",
                    "model_detection": {"type": "prefix"},
                    "pricing": {
                        "input_field": "pricing.prompt",
                        "output_field": "pricing.completion",
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                }
            }
        }
        providers_file = tmp_path / "providers.yaml"
        with open(providers_file, "w") as f:
            import yaml

            yaml.dump(providers_content, f)

        # Create minimal config.yaml
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"model_list": []}, f)

        class TestCleaner(ConfigDrivenModelCleaner):
            pass

        cleaner = TestCleaner(
            provider_name="test_provider",
            config_path=str(config_file),
            providers_config_path=str(providers_file),
        )

        assert cleaner.PROVIDER_NAME == "TestProvider"
        assert cleaner.API_URL == "https://api.test.com/models"
        assert cleaner.MODEL_PREFIX == "test/"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
