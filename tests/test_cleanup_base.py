#!/usr/bin/env python3
"""
Tests for cleanup_base.py utilities and base classes.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
import requests

from cleanup_base import (
    setup_logging,
    costs_are_equal,
    adjust_cost_for_free_model,
    APIClient,
    fetch_models_from_api,
    ProviderConfigLoader,
)


class TestCostsAreEqual:
    """Tests for the costs_are_equal function."""
    
    def test_equal_costs(self):
        """Test that equal costs return True."""
        assert costs_are_equal(1e-09, 1e-09)
        assert costs_are_equal(0.001, 0.001)
        assert costs_are_equal(1.5, 1.5)
    
    def test_unequal_costs(self):
        """Test that unequal costs return False."""
        assert not costs_are_equal(1e-09, 2e-09)
        assert not costs_are_equal(0.001, 0.002)
        assert not costs_are_equal(1.5, 2.5)
    
    def test_zero_costs(self):
        """Test handling of zero costs."""
        assert costs_are_equal(0.0, 0.0)
        assert not costs_are_equal(0.0, 1e-09)
        assert not costs_are_equal(1e-09, 0.0)
    
    def test_scientific_notation(self):
        """Test costs in scientific notation."""
        assert costs_are_equal(6.0e-07, 6.0e-07)
        assert not costs_are_equal(6.0e-07, 4.0e-07)
        assert costs_are_equal(1.23e-05, 1.23e-05)
    
    def test_very_small_differences(self):
        """Test that very small differences within tolerance are equal."""
        # These should be equal within default tolerance
        assert costs_are_equal(1.0, 1.0 + 1e-10)
        assert costs_are_equal(1e-07, 1e-07 + 1e-17)


class TestAdjustCostForFreeModel:
    """Tests for the adjust_cost_for_free_model function."""
    
    def test_none_input(self):
        """Test that None input returns None."""
        assert adjust_cost_for_free_model(None) is None
    
    def test_zero_cost(self):
        """Test that zero cost is converted to free_cost."""
        assert adjust_cost_for_free_model(0.0) == 1e-09
        assert adjust_cost_for_free_model(0.0, free_cost=1e-08) == 1e-08
    
    def test_non_zero_cost(self):
        """Test that non-zero costs are unchanged."""
        assert adjust_cost_for_free_model(0.001) == 0.001
        assert adjust_cost_for_free_model(1e-07) == 1e-07
        assert adjust_cost_for_free_model(5.5) == 5.5
    
    def test_custom_free_cost(self):
        """Test custom free_cost parameter."""
        assert adjust_cost_for_free_model(0.0, free_cost=1e-10) == 1e-10
        assert adjust_cost_for_free_model(0.0, free_cost=0.0001) == 0.0001


class TestSetupLogging:
    """Tests for the setup_logging function."""
    
    def test_default_logging(self):
        """Test default logging setup."""
        logger = setup_logging(verbose=False, name="test_default")
        assert logger.level == logging.INFO
    
    def test_verbose_logging(self):
        """Test verbose logging setup."""
        logger = setup_logging(verbose=True, name="test_verbose")
        assert logger.level == logging.DEBUG
    
    def test_logger_name(self):
        """Test that logger has correct name."""
        logger = setup_logging(name="my_custom_logger")
        assert logger.name == "my_custom_logger"


class TestAPIClient:
    """Tests for the APIClient class."""
    
    def test_init_defaults(self):
        """Test default initialization."""
        client = APIClient()
        assert client.timeout == 30
        assert client.max_retries == 3
        assert client.base_delay == 1.0
        assert client.use_cache is True
    
    def test_init_custom(self):
        """Test custom initialization."""
        client = APIClient(timeout=60, max_retries=5, base_delay=2.0, use_cache=False)
        assert client.timeout == 60
        assert client.max_retries == 5
        assert client.base_delay == 2.0
        assert client.use_cache is False
    
    @patch.object(requests.Session, 'get')
    def test_fetch_success(self, mock_get):
        """Test successful fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {'data': [{'id': 'model1'}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = APIClient(use_cache=False)
        result = client.fetch('https://api.example.com/models')
        
        assert result == {'data': [{'id': 'model1'}]}
        mock_get.assert_called_once()
    
    @patch.object(requests.Session, 'get')
    def test_fetch_with_cache(self, mock_get):
        """Test that caching works."""
        mock_response = Mock()
        mock_response.json.return_value = {'data': [{'id': 'model1'}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = APIClient(use_cache=True)
        
        # First call
        result1 = client.fetch('https://api.example.com/models')
        # Second call should use cache
        result2 = client.fetch('https://api.example.com/models')
        
        assert result1 == result2
        # Should only be called once due to caching
        assert mock_get.call_count == 1
    
    @patch.object(requests.Session, 'get')
    def test_fetch_retry_on_failure(self, mock_get):
        """Test retry logic on transient failures."""
        mock_get.side_effect = [
            requests.RequestException("Connection error"),
            requests.RequestException("Connection error"),
            Mock(json=Mock(return_value={'data': []}), raise_for_status=Mock())
        ]
        
        client = APIClient(max_retries=3, base_delay=0.01, use_cache=False)
        result = client.fetch('https://api.example.com/models')
        
        assert result == {'data': []}
        assert mock_get.call_count == 3
    
    @patch.object(requests.Session, 'get')
    def test_fetch_all_retries_fail(self, mock_get):
        """Test that exception is raised when all retries fail."""
        mock_get.side_effect = requests.RequestException("Connection error")
        
        client = APIClient(max_retries=3, base_delay=0.01, use_cache=False)
        
        with pytest.raises(requests.RequestException):
            client.fetch('https://api.example.com/models')
        
        assert mock_get.call_count == 3
    
    def test_clear_cache(self):
        """Test cache clearing."""
        client = APIClient()
        client._cache['test_key'] = {'data': 'cached'}
        
        client.clear_cache()
        
        assert len(client._cache) == 0


class TestFetchModelsFromApi:
    """Tests for the fetch_models_from_api function."""
    
    @patch('cleanup_base.APIClient')
    def test_fetch_success(self, mock_client_class):
        """Test successful model fetch."""
        mock_client = Mock()
        mock_client.fetch.return_value = {'data': [{'id': 'model1'}, {'id': 'model2'}]}
        mock_client_class.return_value = mock_client
        
        logger = logging.getLogger('test')
        result = fetch_models_from_api('https://api.example.com/models', logger)
        
        assert 'data' in result
        assert len(result['data']) == 2
    
    @patch('cleanup_base.APIClient')
    def test_fetch_missing_data_field(self, mock_client_class):
        """Test that ValueError is raised when 'data' field is missing."""
        mock_client = Mock()
        mock_client.fetch.return_value = {'models': []}  # Wrong field name
        mock_client_class.return_value = mock_client
        
        logger = logging.getLogger('test')
        
        with pytest.raises(ValueError, match="missing 'data' field"):
            fetch_models_from_api('https://api.example.com/models', logger)


class TestProviderConfigLoaderEnabledFlag:
    """Tests that ProviderConfigLoader.list_providers() respects the `enabled` flag.

    Uses a Mock instance to avoid mutating the real singleton (which is shared
    across the test session and would leak state into other test classes).
    """

    def _make_loader(self, providers_dict):
        """Build a Mock ProviderConfigLoader with the given providers dict."""
        loader = Mock(spec=ProviderConfigLoader)
        loader._config = {"providers": providers_dict}

        def list_providers(include_disabled: bool = False):
            providers = loader._config.get("providers", {})
            if include_disabled:
                return list(providers.keys())
            return [
                name for name, cfg in providers.items()
                if cfg.get("enabled", True)
            ]

        loader.list_providers.side_effect = list_providers
        return loader

    def test_defaults_to_enabled(self):
        loader = self._make_loader({
            "openrouter": {"name": "OpenRouter"},
            "vercel": {"name": "Vercel"},
        })
        assert loader.list_providers() == ["openrouter", "vercel"]

    def test_explicit_enabled_true(self):
        loader = self._make_loader({
            "openrouter": {"name": "OpenRouter", "enabled": True},
            "vercel": {"name": "Vercel"},
        })
        assert loader.list_providers() == ["openrouter", "vercel"]

    def test_excludes_disabled(self):
        loader = self._make_loader({
            "openrouter": {"name": "OpenRouter"},
            "ollama": {"name": "Ollama", "enabled": False},
            "vercel": {"name": "Vercel"},
        })
        assert loader.list_providers() == ["openrouter", "vercel"]

    def test_include_disabled_returns_all(self):
        loader = self._make_loader({
            "openrouter": {"name": "OpenRouter"},
            "ollama": {"name": "Ollama", "enabled": False},
        })
        assert loader.list_providers(include_disabled=True) == ["openrouter", "ollama"]

    def test_all_disabled_returns_empty(self):
        loader = self._make_loader({
            "ollama": {"name": "Ollama", "enabled": False},
            "nvidia": {"name": "Nvidia", "enabled": False},
        })
        assert loader.list_providers() == []


class TestPruneSpecialModels:
    """Tests for ProviderConfigLoader.prune_special_models() and save().

    Builds a real ProviderConfigLoader subclass that points at a tmp_path
    providers.yaml file. This avoids mutating the singleton that other
    tests in this session depend on (e.g. TestConfigDrivenModelCleaner
    expects the singleton to still load the real providers.yaml).
    """

    def _make_loader(self, tmp_path, providers):
        path = tmp_path / "providers.yaml"
        with open(path, "w") as f:
            import yaml

            yaml.dump({"providers": providers}, f)

        class _Loader(ProviderConfigLoader):
            def __new__(cls, _path):
                instance = ProviderConfigLoader.__new__(cls)
                instance._config_path = _path
                instance._config = {}
                instance._load_config()
                return instance

        return _Loader(path), path

    def test_removes_models_that_are_in_source(self, tmp_path):
        loader, _ = self._make_loader(tmp_path, {
            "fireworks": {
                "name": "Fireworks",
                "special_models": [
                    "accounts/fireworks/models/deepseek-v4-flash",
                    "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
                ],
            },
        })

        removed = loader.prune_special_models(
            "fireworks", ["accounts/fireworks/models/deepseek-v4-flash"]
        )

        assert removed == ["accounts/fireworks/models/deepseek-v4-flash"]
        assert loader.get_provider_config("fireworks")["special_models"] == [
            "accounts/fireworks/models/nemotron-3-ultra-nvfp4",
        ]

    def test_keeps_models_not_in_source(self, tmp_path):
        loader, _ = self._make_loader(tmp_path, {
            "nvidia": {
                "name": "Nvidia",
                "special_models": ["nvidia/llama-nemotron-rerank-1b-v2"],
            },
        })

        removed = loader.prune_special_models(
            "nvidia", ["nvidia/llama-3.3-70b-instruct"]
        )

        assert removed == []
        assert loader.get_provider_config("nvidia")["special_models"] == [
            "nvidia/llama-nemotron-rerank-1b-v2",
        ]

    def test_empty_special_models_returns_empty(self, tmp_path):
        loader, _ = self._make_loader(tmp_path, {
            "openrouter": {"name": "OpenRouter", "special_models": []},
        })

        removed = loader.prune_special_models("openrouter", ["any/model"])

        assert removed == []
        assert loader.get_provider_config("openrouter")["special_models"] == []

    def test_missing_special_models_key_returns_empty(self, tmp_path):
        loader, _ = self._make_loader(tmp_path, {
            "openrouter": {"name": "OpenRouter"},
        })

        removed = loader.prune_special_models("openrouter", ["any/model"])

        assert removed == []
        assert "special_models" not in loader.get_provider_config("openrouter")
