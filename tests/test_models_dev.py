#!/usr/bin/env python3
"""
Tests for ModelsDevClient and models.dev cost augmentation integration.

Tests cover:
- ModelsDevClient cost extraction and per-million → per-token conversion
- Missing provider / missing model handling
- Network failure graceful fallback
- Integration with ConfigDrivenModelCleaner.parse_api_model fallback
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import requests

from cleanup_base import (
    ModelsDevClient,
    ConfigDrivenModelCleaner,
    _models_dev_client,
    costs_are_equal,
)


# Sample models.dev API response structure
SAMPLE_MODELS_DEV_DATA = {
    "fireworks-ai": {
        "id": "fireworks-ai",
        "name": "Fireworks AI",
        "models": {
            "accounts/fireworks/models/deepseek-v4-pro": {
                "id": "accounts/fireworks/models/deepseek-v4-pro",
                "name": "DeepSeek V4 Pro",
                "cost": {
                    "input": 1.74,
                    "output": 3.48,
                    "cache_read": 0.15,
                },
            },
            "accounts/fireworks/models/glm-5": {
                "id": "accounts/fireworks/models/glm-5",
                "name": "GLM-5",
                "cost": {
                    "input": 1.0,
                    "output": 3.2,
                },
            },
            "accounts/fireworks/models/free-model": {
                "id": "accounts/fireworks/models/free-model",
                "name": "Free Model",
                "cost": {
                    "input": 0,
                    "output": 0,
                },
            },
        },
    },
    "opencode": {
        "id": "opencode",
        "name": "OpenCode Zen",
        "models": {
            "claude-sonnet-4-6": {
                "id": "claude-sonnet-4-6",
                "name": "Claude Sonnet 4.6",
                "cost": {
                    "input": 3,
                    "output": 15,
                },
            },
        },
    },
    "opencode-go": {
        "id": "opencode-go",
        "name": "OpenCode Go",
        "models": {
            "deepseek-v4-pro": {
                "id": "deepseek-v4-pro",
                "name": "DeepSeek V4 Pro",
                "cost": {
                    "input": 1.74,
                    "output": 3.48,
                },
            },
        },
    },
    "empty-provider": {
        "id": "empty-provider",
        "name": "Empty Provider",
        "models": {},
    },
    "no-cost-provider": {
        "id": "no-cost-provider",
        "name": "No Cost Provider",
        "models": {
            "model-no-cost": {
                "id": "model-no-cost",
                "name": "Model Without Cost",
                # No "cost" field
            },
        },
    },
}


class TestModelsDevClient:
    """Tests for the ModelsDevClient class."""

    def test_init(self):
        """Test default initialization."""
        client = ModelsDevClient()
        assert client._data is None
        assert client._load_failed is False

    def test_get_model_cost_fireworks(self):
        """Test cost extraction for a Fireworks model."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "fireworks-ai", "accounts/fireworks/models/deepseek-v4-pro"
        )

        # 1.74 per million → 1.74e-06 per token
        assert input_cost == pytest.approx(1.74e-06)
        # 3.48 per million → 3.48e-06 per token
        assert output_cost == pytest.approx(3.48e-06)

    def test_get_model_cost_opencode_zen(self):
        """Test cost extraction for an OpenCode Zen model."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "opencode", "claude-sonnet-4-6"
        )

        assert input_cost == pytest.approx(3e-06)
        assert output_cost == pytest.approx(15e-06)

    def test_get_model_cost_opencode_go(self):
        """Test cost extraction for an OpenCode Go model."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "opencode-go", "deepseek-v4-pro"
        )

        assert input_cost == pytest.approx(1.74e-06)
        assert output_cost == pytest.approx(3.48e-06)

    def test_get_model_cost_free_model(self):
        """Test cost extraction for a free model (0 cost)."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "fireworks-ai", "accounts/fireworks/models/free-model"
        )

        assert input_cost == pytest.approx(0.0)
        assert output_cost == pytest.approx(0.0)

    def test_get_model_cost_missing_provider(self):
        """Test that missing provider returns (None, None)."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "nonexistent-provider", "some-model"
        )

        assert input_cost is None
        assert output_cost is None

    def test_get_model_cost_missing_model(self):
        """Test that missing model returns (None, None)."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "fireworks-ai", "accounts/fireworks/models/nonexistent-model"
        )

        assert input_cost is None
        assert output_cost is None

    def test_get_model_cost_no_cost_field(self):
        """Test model without a cost field returns (None, None)."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "no-cost-provider", "model-no-cost"
        )

        assert input_cost is None
        assert output_cost is None

    def test_get_model_cost_empty_provider(self):
        """Test provider with no models returns (None, None)."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        input_cost, output_cost = client.get_model_cost(
            "empty-provider", "any-model"
        )

        assert input_cost is None
        assert output_cost is None

    def test_get_model_cost_data_not_loaded(self):
        """Test that unloaded data returns (None, None) when load fails."""
        client = ModelsDevClient()
        client._load_failed = True  # Simulate failed load

        input_cost, output_cost = client.get_model_cost(
            "fireworks-ai", "accounts/fireworks/models/deepseek-v4-pro"
        )

        assert input_cost is None
        assert output_cost is None

    def test_ensure_loaded_success(self):
        """Test successful data loading."""
        client = ModelsDevClient()
        client._api_client = Mock()
        client._api_client.fetch.return_value = SAMPLE_MODELS_DEV_DATA

        logger = logging.getLogger("test_ensure_loaded")
        client._ensure_loaded(logger)

        assert client._data == SAMPLE_MODELS_DEV_DATA
        assert client._load_failed is False
        client._api_client.fetch.assert_called_once()

    def test_ensure_loaded_network_failure(self):
        """Test graceful handling of network failure."""
        client = ModelsDevClient()
        client._api_client = Mock()
        client._api_client.fetch.side_effect = requests.RequestException("Connection error")

        logger = logging.getLogger("test_network_failure")
        client._ensure_loaded(logger)

        assert client._data is None
        assert client._load_failed is True

    def test_ensure_loaded_only_once(self):
        """Test that data is only fetched once (cached)."""
        client = ModelsDevClient()
        client._api_client = Mock()
        client._api_client.fetch.return_value = SAMPLE_MODELS_DEV_DATA

        client._ensure_loaded()
        client._ensure_loaded()  # Second call should be a no-op

        client._api_client.fetch.assert_called_once()

    def test_ensure_loaded_no_retry_after_failure(self):
        """Test that failed load is not retried."""
        client = ModelsDevClient()
        client._api_client = Mock()
        client._api_client.fetch.side_effect = requests.RequestException("Error")

        client._ensure_loaded()
        assert client._load_failed is True

        # Reset the mock to succeed
        client._api_client.fetch.side_effect = None
        client._api_client.fetch.return_value = SAMPLE_MODELS_DEV_DATA

        client._ensure_loaded()  # Should not retry
        assert client._data is None  # Still None

    def test_clear_cache(self):
        """Test cache clearing allows fresh fetch."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA
        client._load_failed = False

        client.clear_cache()

        assert client._data is None
        assert client._load_failed is False

    def test_clear_cache_resets_failure(self):
        """Test that clear_cache resets failure flag."""
        client = ModelsDevClient()
        client._load_failed = True

        client.clear_cache()

        assert client._load_failed is False

    def test_conversion_precision(self):
        """Test that per-million to per-token conversion maintains precision."""
        client = ModelsDevClient()
        client._data = {
            "test-provider": {
                "models": {
                    "test-model": {
                        "cost": {
                            "input": 0.29,
                            "output": 0.43,
                        }
                    }
                }
            }
        }

        input_cost, output_cost = client.get_model_cost(
            "test-provider", "test-model"
        )

        # 0.29 / 1_000_000 = 2.9e-07
        assert input_cost == pytest.approx(2.9e-07)
        # 0.43 / 1_000_000 = 4.3e-07
        assert output_cost == pytest.approx(4.3e-07)


class TestParseApiModelWithModelsDevFallback:
    """Tests for ConfigDrivenModelCleaner.parse_api_model with models.dev fallback."""

    def _create_cleaner_with_mock(self, models_dev_id=None, models_dev_data=None):
        """Helper to create a cleaner with mocked models.dev data."""
        # We need to mock ProviderConfigLoader to avoid needing a real providers.yaml
        with patch("cleanup_base.ProviderConfigLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_provider_config.return_value = {
                "name": "Test Provider",
                "api_url": "https://api.test.com/models",
                "model_prefix": "test/",
                "model_detection": {"type": "prefix", "value": "test/"},
                "pricing": {
                    "input_field": None,
                    "output_field": None,
                    "is_per_million": False,
                    "free_model_handling": True,
                    "default_cost": None,
                    "models_dev_id": models_dev_id,
                },
                "model_name_prefix": "",
                "model_name_cleanup": [],
                "special_models": [],
                "order": 4,
            }
            mock_loader_class.return_value = mock_loader

            cleaner = ConfigDrivenModelCleaner(
                "test_provider", "config.yaml", dry_run=True, verbose=True
            )

        return cleaner

    @patch("cleanup_base._models_dev_client")
    def test_fallback_to_models_dev_when_no_pricing(self, mock_client):
        """Test that models.dev is used when provider API has no pricing."""
        mock_client.get_model_cost.return_value = (1.74e-06, 3.48e-06)

        cleaner = self._create_cleaner_with_mock(models_dev_id="fireworks-ai")

        model = {"id": "accounts/fireworks/models/deepseek-v4-pro"}
        result = cleaner.parse_api_model(model)

        assert result["input_cost"] == pytest.approx(1.74e-06)
        assert result["output_cost"] == pytest.approx(3.48e-06)
        mock_client.get_model_cost.assert_called_once_with(
            "fireworks-ai", "accounts/fireworks/models/deepseek-v4-pro",
            cleaner.logger,
        )

    @patch("cleanup_base._models_dev_client")
    def test_no_fallback_when_no_models_dev_id(self, mock_client):
        """Test that models.dev is NOT used when models_dev_id is not configured."""
        cleaner = self._create_cleaner_with_mock(models_dev_id=None)

        model = {"id": "some-model"}
        result = cleaner.parse_api_model(model)

        assert result["input_cost"] is None
        assert result["output_cost"] is None
        mock_client.get_model_cost.assert_not_called()

    @patch("cleanup_base._models_dev_client")
    def test_fallback_returns_none_for_unknown_model(self, mock_client):
        """Test fallback returns None costs when model not found in models.dev."""
        mock_client.get_model_cost.return_value = (None, None)

        cleaner = self._create_cleaner_with_mock(models_dev_id="fireworks-ai")

        model = {"id": "accounts/fireworks/models/unknown-model"}
        result = cleaner.parse_api_model(model)

        assert result["input_cost"] is None
        assert result["output_cost"] is None

    @patch("cleanup_base._models_dev_client")
    def test_no_fallback_when_provider_has_pricing_fields(self, mock_client):
        """Test that models.dev is NOT used when provider pricing fields return costs."""
        with patch("cleanup_base.ProviderConfigLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_provider_config.return_value = {
                "name": "Test Provider With Pricing",
                "api_url": "https://api.test.com/models",
                "model_prefix": "test/",
                "model_detection": {"type": "prefix", "value": "test/"},
                "pricing": {
                    "input_field": "pricing.prompt",
                    "output_field": "pricing.completion",
                    "is_per_million": False,
                    "free_model_handling": True,
                    "default_cost": None,
                    "models_dev_id": "some-provider",  # Even if configured
                },
                "model_name_prefix": "",
                "model_name_cleanup": [],
                "special_models": [],
                "order": 5,
            }
            mock_loader_class.return_value = mock_loader

            cleaner = ConfigDrivenModelCleaner(
                "test_with_pricing", "config.yaml", dry_run=True, verbose=True
            )

        # Model with pricing from provider API
        model = {
            "id": "test-model",
            "pricing": {"prompt": "0.000001", "completion": "0.000003"},
        }
        result = cleaner.parse_api_model(model)

        # Should use provider API pricing, not models.dev
        assert result["input_cost"] is not None
        assert result["output_cost"] is not None
        mock_client.get_model_cost.assert_not_called()

    @patch("cleanup_base._models_dev_client")
    def test_no_fallback_when_default_cost_set(self, mock_client):
        """Test that models.dev is NOT used when default_cost is set (free providers)."""
        with patch("cleanup_base.ProviderConfigLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_provider_config.return_value = {
                "name": "Free Provider",
                "api_url": "https://api.test.com/models",
                "model_prefix": "test/",
                "model_detection": {"type": "prefix", "value": "test/"},
                "pricing": {
                    "input_field": None,
                    "output_field": None,
                    "is_per_million": False,
                    "free_model_handling": True,
                    "default_cost": 1.0e-09,
                    "models_dev_id": "some-provider",  # Even if configured
                },
                "model_name_prefix": "",
                "model_name_cleanup": [],
                "special_models": [],
                "order": 5,
            }
            mock_loader_class.return_value = mock_loader

            cleaner = ConfigDrivenModelCleaner(
                "free_provider", "config.yaml", dry_run=True, verbose=True
            )

        model = {"id": "free-model"}
        result = cleaner.parse_api_model(model)

        # Should use default_cost, not models.dev
        assert result["input_cost"] == 1.0e-09
        assert result["output_cost"] == 1.0e-09
        mock_client.get_model_cost.assert_not_called()


class TestModelsDevClientGetProviderModels:
    """Tests for ModelsDevClient.get_provider_models()."""

    def test_returns_models_with_per_token_costs(self):
        """Per-million costs from models.dev are converted to per-token costs."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        models = client.get_provider_models("fireworks-ai")

        assert (
            models["accounts/fireworks/models/deepseek-v4-pro"]["input_cost"]
            == pytest.approx(1.74e-06)
        )
        assert (
            models["accounts/fireworks/models/deepseek-v4-pro"]["output_cost"]
            == pytest.approx(3.48e-06)
        )
        assert (
            models["accounts/fireworks/models/glm-5"]["input_cost"]
            == pytest.approx(1.0e-06)
        )

    def test_returns_id_and_none_model_info(self):
        """Each entry exposes `id` matching the key and `model_info` is None."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        models = client.get_provider_models("fireworks-ai")

        entry = models["accounts/fireworks/models/deepseek-v4-pro"]
        assert entry["id"] == "accounts/fireworks/models/deepseek-v4-pro"
        assert entry["model_info"] is None

    def test_free_model_zero_cost_returned(self):
        """A model with zero cost returns 0.0 (free_model_handling is applied later)."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        models = client.get_provider_models("fireworks-ai")

        entry = models["accounts/fireworks/models/free-model"]
        assert entry["input_cost"] == pytest.approx(0.0)
        assert entry["output_cost"] == pytest.approx(0.0)

    def test_raises_when_data_unavailable(self):
        """A RuntimeError is raised when models.dev data could not be loaded."""
        client = ModelsDevClient()
        client._load_failed = True

        with pytest.raises(RuntimeError):
            client.get_provider_models("fireworks-ai")

    def test_raises_for_missing_provider(self):
        """A ValueError is raised when the provider is not in models.dev data."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        with pytest.raises(ValueError):
            client.get_provider_models("nonexistent-provider")

    def test_skips_malformed_model_entries(self):
        """Non-dict model entries are ignored without raising."""
        client = ModelsDevClient()
        client._data = {
            "test-provider": {
                "models": {
                    "good-model": {"id": "good-model", "cost": {"input": 1.0, "output": 2.0}},
                    "not-a-dict": "oops",
                }
            }
        }

        models = client.get_provider_models("test-provider")

        assert "good-model" in models
        assert "not-a-dict" not in models

    def test_empty_models_returns_empty_dict(self):
        """A provider with no models returns an empty dict (not an error)."""
        client = ModelsDevClient()
        client._data = SAMPLE_MODELS_DEV_DATA

        models = client.get_provider_models("empty-provider")

        assert models == {}


class TestFetchAvailableModelsFromModelsDev:
    """Tests for ConfigDrivenModelCleaner.fetch_available_models() when
    use_models_dev_for_listing is enabled."""

    def _create_cleaner(self, **overrides):
        """Helper to create a cleaner with use_models_dev_for_listing enabled."""
        provider_config = {
            "name": "Fireworks",
            "api_url": "https://api.fireworks.ai/inference/v1/models",
            "model_prefix": "fireworks_ai/",
            "model_detection": {"type": "prefix", "value": "fireworks_ai/"},
            "pricing": {
                "input_field": None,
                "output_field": None,
                "is_per_million": False,
                "free_model_handling": True,
                "default_cost": None,
                "models_dev_id": "fireworks-ai",
            },
            "model_name_prefix": "",
            "model_name_cleanup": [],
            "special_models": [
                "accounts/fireworks/models/qwen3-embedding-8b",
            ],
            "order": 4,
        }
        provider_config["use_models_dev_for_listing"] = True
        provider_config.update(overrides)

        with patch("cleanup_base.ProviderConfigLoader") as mock_loader_class:
            mock_loader = Mock()
            mock_loader.get_provider_config.return_value = provider_config
            mock_loader_class.return_value = mock_loader

            cleaner = ConfigDrivenModelCleaner(
                "fireworks", "config.yaml", dry_run=True, verbose=True
            )
        return cleaner

    @patch("cleanup_base._models_dev_client")
    def test_uses_models_dev_when_flag_enabled(self, mock_client):
        """fetch_available_models pulls the catalog from models.dev."""
        mock_client.get_provider_models.return_value = {
            "accounts/fireworks/models/deepseek-v4-pro": {
                "id": "accounts/fireworks/models/deepseek-v4-pro",
                "input_cost": 1.74e-06,
                "output_cost": 3.48e-06,
                "model_info": None,
            }
        }

        cleaner = self._create_cleaner()
        result = cleaner.fetch_available_models()

        assert "accounts/fireworks/models/deepseek-v4-pro" in result
        assert (
            result["accounts/fireworks/models/deepseek-v4-pro"]["input_cost"]
            == pytest.approx(1.74e-06)
        )
        mock_client.get_provider_models.assert_called_once_with(
            "fireworks-ai", cleaner.logger
        )

    @patch("cleanup_base._models_dev_client")
    def test_applies_free_model_handling(self, mock_client):
        """Zero costs from models.dev are converted when free_model_handling is true."""
        mock_client.get_provider_models.return_value = {
            "accounts/fireworks/models/free-model": {
                "id": "accounts/fireworks/models/free-model",
                "input_cost": 0.0,
                "output_cost": 0.0,
                "model_info": None,
            }
        }

        cleaner = self._create_cleaner()
        result = cleaner.fetch_available_models()

        assert (
            result["accounts/fireworks/models/free-model"]["input_cost"]
            == pytest.approx(1e-09)
        )
        assert (
            result["accounts/fireworks/models/free-model"]["output_cost"]
            == pytest.approx(1e-09)
        )

    @patch("cleanup_base._models_dev_client")
    def test_does_not_apply_free_handling_when_disabled(self, mock_client):
        """Zero costs are kept as 0.0 when free_model_handling is false."""
        mock_client.get_provider_models.return_value = {
            "accounts/fireworks/models/free-model": {
                "id": "accounts/fireworks/models/free-model",
                "input_cost": 0.0,
                "output_cost": 0.0,
                "model_info": None,
            }
        }

        cleaner = self._create_cleaner(
            pricing={
                "input_field": None,
                "output_field": None,
                "is_per_million": False,
                "free_model_handling": False,
                "default_cost": None,
                "models_dev_id": "fireworks-ai",
            }
        )
        # Re-apply the flag because the overrides may have wiped it
        cleaner._use_models_dev_for_listing = True
        cleaner._pricing_config["free_model_handling"] = False

        result = cleaner.fetch_available_models()

        assert result["accounts/fireworks/models/free-model"]["input_cost"] == 0.0
        assert result["accounts/fireworks/models/free-model"]["output_cost"] == 0.0

    def test_raises_when_models_dev_id_missing(self):
        """A ValueError is raised if models_dev_id is not configured."""
        cleaner = self._create_cleaner(
            pricing={
                "input_field": None,
                "output_field": None,
                "is_per_million": False,
                "free_model_handling": True,
                "default_cost": None,
                "models_dev_id": None,
            }
        )
        cleaner._use_models_dev_for_listing = True
        cleaner._models_dev_id = None

        with pytest.raises(ValueError):
            cleaner.fetch_available_models()

    @patch("cleanup_base._models_dev_client")
    def test_propagates_models_dev_failure(self, mock_client):
        """Failures from models.dev propagate (no fallback to provider api_url)."""
        mock_client.get_provider_models.side_effect = RuntimeError(
            "models.dev data is unavailable"
        )

        cleaner = self._create_cleaner()

        with pytest.raises(RuntimeError):
            cleaner.fetch_available_models()

    @patch("cleanup_base._models_dev_client")
    def test_special_models_protected_during_validation(self, mock_client):
        """special_models are not flagged as invalid when sourced from models.dev."""
        mock_client.get_provider_models.return_value = {
            "accounts/fireworks/models/known-model": {
                "id": "accounts/fireworks/models/known-model",
                "input_cost": 1.0e-06,
                "output_cost": 2.0e-06,
                "model_info": None,
            }
        }

        cleaner = self._create_cleaner()
        api_models = cleaner.fetch_available_models()

        # First entry is the special model — not present in models.dev listing
        config_models = [
            (0, "fireworks_ai/accounts/fireworks/models/qwen3-embedding-8b", "emb"),
            (1, "fireworks_ai/accounts/fireworks/models/known-model", "known"),
        ]

        invalid = cleaner.validate_models(config_models, api_models)
        assert invalid == []

    @patch("cleanup_base._models_dev_client")
    def test_does_not_call_provider_api(self, mock_client):
        """The provider's api_url is NOT called when use_models_dev_for_listing is true."""
        mock_client.get_provider_models.return_value = {}

        cleaner = self._create_cleaner()

        with patch.object(cleaner, "_build_api_headers", return_value=None), patch(
            "cleanup_base.fetch_models_from_api"
        ) as mock_fetch:
            cleaner.fetch_available_models()
            mock_fetch.assert_not_called()

    @patch("cleanup_base._models_dev_client")
    def test_embedding_endpoint_does_not_overwrite_existing(self, mock_client):
        """Embedding endpoint adds new IDs but never overwrites models.dev entries."""
        mock_client.get_provider_models.return_value = {
            "accounts/fireworks/models/chat-model": {
                "id": "accounts/fireworks/models/chat-model",
                "input_cost": 1.0e-06,
                "output_cost": 2.0e-06,
                "model_info": None,
            },
            "accounts/fireworks/models/shared-id": {
                "id": "accounts/fireworks/models/shared-id",
                "input_cost": 1.5e-06,
                "output_cost": 2.5e-06,
                "model_info": None,
            },
        }
        # Avoid the models.dev fallback inside parse_api_model by giving it
        # a (no-op) tuple return value.
        mock_client.get_model_cost.return_value = (None, None)

        cleaner = self._create_cleaner(
            embeddings_api_url="https://example.com/embeddings",
            pricing={
                "input_field": "pricing.prompt",
                "output_field": "pricing.completion",
                "is_per_million": False,
                "free_model_handling": True,
                "default_cost": None,
                "models_dev_id": "fireworks-ai",
            },
        )
        cleaner._embeddings_api_url = "https://example.com/embeddings"

        embed_response = {
            "data": [
                {
                    "id": "accounts/fireworks/models/shared-id",
                    "pricing": {"prompt": "0.000009", "completion": "0.000008"},
                },
                {
                    "id": "accounts/fireworks/models/embedding-only",
                    "pricing": {"prompt": "0.000001", "completion": "0.000001"},
                },
            ]
        }

        with patch.object(cleaner, "_build_api_headers", return_value=None), patch(
            "cleanup_base.fetch_models_from_api", return_value=embed_response
        ):
            result = cleaner.fetch_available_models()

        # The chat model from models.dev was not overwritten
        assert (
            result["accounts/fireworks/models/shared-id"]["input_cost"]
            == pytest.approx(1.5e-06)
        )
        # The new embedding-only model was added
        assert "accounts/fireworks/models/embedding-only" in result
