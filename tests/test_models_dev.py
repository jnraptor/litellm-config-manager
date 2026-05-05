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
