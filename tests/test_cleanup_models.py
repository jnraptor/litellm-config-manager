#!/usr/bin/env python3
"""
Tests for cleanup_models.py unified cleanup script.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import yaml

from cleanup_models import (
    ProviderConfig,
    ProviderManager,
    PrefixDetectionStrategy,
    ApiBaseDetectionStrategy,
    UnifiedModelCleaner,
)


class TestProviderConfig:
    """Tests for ProviderConfig dataclass validation."""
    
    def test_valid_prefix_detection(self):
        """Test valid prefix detection config."""
        config = ProviderConfig(
            name="TestProvider",
            description="Test provider",
            api_url="https://api.test.com/models",
            model_prefix="test/",
            model_detection={'type': 'prefix', 'value': 'test/'},
            pricing={'input_field': 'prompt', 'output_field': 'completion'},
            model_name_prefix="t-",
        )
        assert config.name == "TestProvider"
        assert config.model_detection['type'] == 'prefix'
    
    def test_valid_api_base_detection(self):
        """Test valid api_base detection config."""
        config = ProviderConfig(
            name="TestProvider",
            description="Test provider",
            api_url="https://api.test.com/models",
            model_prefix="openai/",
            model_detection={'type': 'api_base', 'value': 'api.test.com'},
            pricing={'input_field': 'input_price', 'output_field': 'output_price'},
            model_name_prefix="",
        )
        assert config.model_detection['type'] == 'api_base'
    
    def test_invalid_detection_type(self):
        """Test that invalid detection type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model_detection type"):
            ProviderConfig(
                name="TestProvider",
                description="Test provider",
                api_url="https://api.test.com/models",
                model_prefix="test/",
                model_detection={'type': 'invalid_type'},
                pricing={},
                model_name_prefix="",
            )
    
    def test_api_base_missing_value(self):
        """Test that api_base detection without value raises ValueError."""
        with pytest.raises(ValueError, match="missing 'value' field"):
            ProviderConfig(
                name="TestProvider",
                description="Test provider",
                api_url="https://api.test.com/models",
                model_prefix="openai/",
                model_detection={'type': 'api_base'},  # Missing 'value'
                pricing={},
                model_name_prefix="",
            )
    
    def test_none_lists_initialized(self):
        """Test that None lists are initialized to empty lists."""
        config = ProviderConfig(
            name="TestProvider",
            description="Test provider",
            api_url="https://api.test.com/models",
            model_prefix="test/",
            model_detection={'type': 'prefix'},
            pricing={},
            model_name_prefix="",
            model_name_cleanup=None,
            special_models=None,
        )
        assert config.model_name_cleanup == []
        assert config.special_models == []


class TestPrefixDetectionStrategy:
    """Tests for PrefixDetectionStrategy."""
    
    @pytest.fixture
    def strategy(self):
        config = ProviderConfig(
            name="OpenRouter",
            description="Test",
            api_url="https://api.test.com",
            model_prefix="openrouter/",
            model_detection={'type': 'prefix'},
            pricing={'input_field': 'prompt', 'output_field': 'completion'},
            model_name_prefix="or-",
            model_name_cleanup=[{'replace': [['anthropic-', '']]}],
        )
        return PrefixDetectionStrategy(config)
    
    def test_is_provider_model_true(self, strategy):
        """Test detection of provider model."""
        model_entry = {
            'model_name': 'test-model',
            'litellm_params': {'model': 'openrouter/anthropic/claude-3'}
        }
        assert strategy.is_provider_model(model_entry) is True
    
    def test_is_provider_model_false(self, strategy):
        """Test non-provider model returns False."""
        model_entry = {
            'model_name': 'test-model',
            'litellm_params': {'model': 'other/model'}
        }
        assert strategy.is_provider_model(model_entry) is False
    
    def test_extract_model_id(self, strategy):
        """Test model ID extraction."""
        model_entry = {
            'litellm_params': {'model': 'openrouter/anthropic/claude-3'}
        }
        assert strategy.extract_model_id(model_entry) == 'anthropic/claude-3'
    
    def test_generate_model_name(self, strategy):
        """Test model name generation with cleanup rules."""
        model_name = strategy.generate_model_name('anthropic-claude-3')
        assert model_name == 'or-claude-3'
    
    def test_parse_api_model(self, strategy):
        """Test API model parsing."""
        api_model = {
            'id': 'test-model',
            'pricing': {'prompt': '0.001', 'completion': '0.002'}
        }
        result = strategy.parse_api_model(api_model)
        
        assert result['id'] == 'test-model'
        assert result['input_cost'] == 0.001
        assert result['output_cost'] == 0.002


class TestApiBaseDetectionStrategy:
    """Tests for ApiBaseDetectionStrategy."""
    
    @pytest.fixture
    def strategy(self):
        config = ProviderConfig(
            name="Requesty",
            description="Test",
            api_url="https://api.test.com",
            model_prefix="openai/",
            model_detection={'type': 'api_base', 'value': 'router.requesty.ai'},
            pricing={'input_field': 'input_price', 'output_field': 'output_price'},
            model_name_prefix="",
        )
        return ApiBaseDetectionStrategy(config)
    
    def test_is_provider_model_true(self, strategy):
        """Test detection of provider model with api_base."""
        model_entry = {
            'model_name': 'test-model',
            'litellm_params': {
                'model': 'openai/gpt-4',
                'api_base': 'https://router.requesty.ai/v1'
            }
        }
        assert strategy.is_provider_model(model_entry) is True
    
    def test_is_provider_model_wrong_api_base(self, strategy):
        """Test that wrong api_base returns False."""
        model_entry = {
            'model_name': 'test-model',
            'litellm_params': {
                'model': 'openai/gpt-4',
                'api_base': 'https://api.openai.com/v1'
            }
        }
        assert strategy.is_provider_model(model_entry) is False
    
    def test_parse_api_model_nested_pricing(self, strategy):
        """Test parsing with nested pricing fields."""
        # Update strategy config for nested pricing
        strategy.config.pricing['input_field'] = 'pricing.prompt'
        strategy.config.pricing['output_field'] = 'pricing.completion'
        strategy.config.pricing['is_per_million'] = True
        strategy.config.pricing['divisor'] = 1
        
        api_model = {
            'id': 'test-model',
            'pricing': {'prompt': 1000000, 'completion': 2000000}
        }
        result = strategy.parse_api_model(api_model)
        
        assert result['id'] == 'test-model'
        assert result['input_cost'] == 1.0
        assert result['output_cost'] == 2.0


class TestProviderManager:
    """Tests for ProviderManager."""
    
    @pytest.fixture
    def temp_providers_yaml(self, tmp_path):
        """Create a temporary providers.yaml file."""
        providers_content = {
            'providers': {
                'test_provider': {
                    'name': 'TestProvider',
                    'description': 'Test provider',
                    'api_url': 'https://api.test.com/models',
                    'model_prefix': 'test/',
                    'model_detection': {'type': 'prefix'},
                    'pricing': {'input_field': 'prompt', 'output_field': 'completion'},
                    'model_name_prefix': 't-',
                    'model_name_cleanup': [],
                    'special_models': [],
                }
            }
        }
        config_file = tmp_path / 'providers.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(providers_content, f)
        return config_file
    
    def test_load_providers(self, temp_providers_yaml):
        """Test loading providers from YAML."""
        manager = ProviderManager(str(temp_providers_yaml))
        
        assert 'test_provider' in manager.list_providers()
        provider = manager.get_provider('test_provider')
        assert provider.name == 'TestProvider'
    
    def test_get_unknown_provider(self, temp_providers_yaml):
        """Test that unknown provider raises ValueError."""
        manager = ProviderManager(str(temp_providers_yaml))
        
        with pytest.raises(ValueError, match="Unknown provider"):
            manager.get_provider('nonexistent')
    
    def test_get_api_client(self, temp_providers_yaml):
        """Test getting shared API client."""
        manager = ProviderManager(str(temp_providers_yaml))
        client = manager.get_api_client()
        
        assert client is not None
        assert client.timeout == 30


class TestUnifiedModelCleaner:
    """Tests for UnifiedModelCleaner."""
    
    @pytest.fixture
    def temp_config_files(self, tmp_path):
        """Create temporary config files."""
        # Create providers.yaml
        providers_content = {
            'providers': {
                'test_provider': {
                    'name': 'TestProvider',
                    'description': 'Test provider',
                    'api_url': 'https://api.test.com/models',
                    'model_prefix': 'test/',
                    'model_detection': {'type': 'prefix'},
                    'pricing': {
                        'input_field': 'prompt',
                        'output_field': 'completion',
                        'free_model_handling': True
                    },
                    'model_name_prefix': 't-',
                    'model_name_cleanup': [],
                    'special_models': ['special-model'],
                }
            }
        }
        providers_file = tmp_path / 'providers.yaml'
        with open(providers_file, 'w') as f:
            yaml.dump(providers_content, f)
        
        # Create config.yaml
        config_content = {
            'model_list': [
                {
                    'model_name': 't-valid-model',
                    'litellm_params': {
                        'model': 'test/valid-model',
                        'input_cost_per_token': 0.001,
                        'output_cost_per_token': 0.002
                    }
                },
                {
                    'model_name': 't-invalid-model',
                    'litellm_params': {
                        'model': 'test/invalid-model',
                        'input_cost_per_token': 0.001,
                        'output_cost_per_token': 0.002
                    }
                },
                {
                    'model_name': 't-special-model',
                    'litellm_params': {
                        'model': 'test/special-model',
                        'input_cost_per_token': 0.001,
                        'output_cost_per_token': 0.002
                    }
                }
            ]
        }
        config_file = tmp_path / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        return {
            'providers': str(providers_file),
            'config': str(config_file),
            'tmp_path': tmp_path
        }
    
    def test_load_config(self, temp_config_files, monkeypatch):
        """Test loading configuration."""
        monkeypatch.chdir(temp_config_files['tmp_path'])
        
        cleaner = UnifiedModelCleaner(
            config_path=temp_config_files['config'],
            provider_names=['test_provider'],
            dry_run=True
        )
        config = cleaner.load_config()
        
        assert 'model_list' in config
        assert len(config['model_list']) == 3
    
    def test_sort_model_list(self, temp_config_files, monkeypatch):
        """Test model list sorting."""
        monkeypatch.chdir(temp_config_files['tmp_path'])
        
        cleaner = UnifiedModelCleaner(
            config_path=temp_config_files['config'],
            provider_names=['test_provider'],
            dry_run=True
        )
        config = cleaner.load_config()
        
        # Add unsorted models
        config['model_list'].insert(0, {
            'model_name': 'z-model',
            'litellm_params': {'model': 'test/z-model'}
        })
        config['model_list'].insert(0, {
            'model_name': 'a-model',
            'litellm_params': {'model': 'test/a-model'}
        })
        
        sorted_config, was_sorted = cleaner.sort_model_list(config)
        
        assert was_sorted is True
        # First model should be 'a-model' after sorting
        assert sorted_config['model_list'][0]['model_name'] == 'a-model'
    
    def test_extract_provider_models(self, temp_config_files, monkeypatch):
        """Test extracting provider models."""
        monkeypatch.chdir(temp_config_files['tmp_path'])
        
        cleaner = UnifiedModelCleaner(
            config_path=temp_config_files['config'],
            provider_names=['test_provider'],
            dry_run=True
        )
        config = cleaner.load_config()
        
        models = cleaner.extract_provider_models(config, 'test_provider')
        
        assert len(models) == 3
        model_ids = [m[1] for m in models]
        assert 'test/valid-model' in model_ids
    
    @patch.object(ProviderManager, 'get_api_client')
    def test_validate_models_identifies_invalid(self, mock_get_client, temp_config_files, monkeypatch):
        """Test that invalid models are identified."""
        monkeypatch.chdir(temp_config_files['tmp_path'])
        
        cleaner = UnifiedModelCleaner(
            config_path=temp_config_files['config'],
            provider_names=['test_provider'],
            dry_run=True
        )
        config = cleaner.load_config()
        config_models = cleaner.extract_provider_models(config, 'test_provider')
        
        # API only has 'valid-model', not 'invalid-model'
        api_models = {
            'valid-model': {'id': 'valid-model', 'input_cost': 0.001, 'output_cost': 0.002}
        }
        
        invalid_models = cleaner.validate_models(config_models, api_models, 'test_provider')
        
        # Should identify 'invalid-model' as invalid
        # 'special-model' should NOT be marked invalid (it's in special_models)
        invalid_ids = [m[1] for m in invalid_models]
        assert 'test/invalid-model' in invalid_ids
        assert 'test/special-model' not in invalid_ids
