#!/usr/bin/env python3
"""
Unified Model Cleanup and Cost Update Script for LiteLLM Config

This script validates models from multiple providers in a LiteLLM config.yaml file
against their current APIs and:
1. Removes any invalid model entries (except special models)
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing
3. Sorts the model list alphabetically
4. Adds new models when requested
5. Supports both regular and embedding models (where applicable)

Supported providers: openrouter, requesty, novita, nano_gpt, vercel, poe, nvidia, synthetic, all

Usage:
    python cleanup_models.py --provider openrouter [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_models.py --provider all [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_models.py --provider requesty --add-model "model1 model2"

Author: Unified script for LiteLLM Config Management
"""

import argparse
import os
import sys
import yaml
import requests
import time
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from dotenv import load_dotenv

from cleanup_base import (
    setup_logging,
    costs_are_equal,
    adjust_cost_for_free_model,
    APIClient,
)


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    name: str
    description: str
    api_url: str
    model_prefix: str
    model_detection: Dict[str, Any]
    pricing: Dict[str, Any]
    model_name_prefix: str
    model_name_cleanup: List[Dict[str, str]] = field(default_factory=list)
    special_models: List[str] = field(default_factory=list)
    api_base_config: Optional[Dict[str, str]] = None
    api_key_env: Optional[str] = None
    embeddings_api_url: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_detection_types = ('prefix', 'api_base')
        detection_type = self.model_detection.get('type')
        if detection_type not in valid_detection_types:
            raise ValueError(
                f"Invalid model_detection type '{detection_type}' for provider '{self.name}'. "
                f"Must be one of: {valid_detection_types}"
            )
        
        if detection_type == 'api_base' and 'value' not in self.model_detection:
            raise ValueError(
                f"Provider '{self.name}' uses api_base detection but missing 'value' field"
            )
        
        # Ensure lists are initialized
        if self.model_name_cleanup is None:
            self.model_name_cleanup = []
        if self.special_models is None:
            self.special_models = []


class ProviderStrategy(ABC):
    """Abstract base class for provider-specific strategies."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def is_provider_model(self, model_entry: Dict[str, Any]) -> bool:
        """Check if a model entry belongs to this provider."""
        pass

    @abstractmethod
    def extract_model_id(self, model_entry: Dict[str, Any]) -> Optional[str]:
        """Extract the model ID from a model entry."""
        pass

    @abstractmethod
    def parse_api_model(self, api_model: Dict[str, Any]) -> Dict[str, Any]:
        """Parse pricing information from API response."""
        pass

    def generate_model_name(self, model_id: str) -> str:
        """Generate a model name for this provider."""
        clean_id = model_id.replace('/', '-').replace(':', '-')
        for cleanup_rule in self.config.model_name_cleanup:
            for replace_old, replace_new in cleanup_rule.get('replace', []):
                clean_id = clean_id.replace(replace_old, replace_new)
        model_name = f"{self.config.model_name_prefix}{clean_id}".lower()
        return model_name

    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new model entry for the config."""
        model_name = self.generate_model_name(model_id)
        input_cost = api_model_info.get('input_cost')
        output_cost = api_model_info.get('output_cost')
        model_info_section = api_model_info.get('model_info')

        default_cost = self.config.pricing.get('default_cost')
        if default_cost is not None:
            if input_cost is None:
                input_cost = default_cost
            if output_cost is None:
                output_cost = default_cost

        if self.config.pricing.get('free_model_handling', False):
            input_cost = adjust_cost_for_free_model(input_cost)
            output_cost = adjust_cost_for_free_model(output_cost)

        model_entry = {
            'model_name': model_name,
            'litellm_params': {
                'model': f"{self.config.model_prefix}{model_id}"
            }
        }

        if self.config.api_base_config:
            if 'url' in self.config.api_base_config:
                model_entry['litellm_params']['api_base'] = self.config.api_base_config['url']
            elif 'url_env' in self.config.api_base_config:
                model_entry['litellm_params']['api_base'] = f"os.environ/{self.config.api_base_config['url_env']}"
            if 'api_key_env' in self.config.api_base_config:
                model_entry['litellm_params']['api_key'] = f"os.environ/{self.config.api_base_config['api_key_env']}"

        if input_cost is not None:
            model_entry['litellm_params']['input_cost_per_token'] = input_cost
        if output_cost is not None:
            model_entry['litellm_params']['output_cost_per_token'] = output_cost
        if model_info_section:
            model_entry['model_info'] = model_info_section

        return model_entry


class PrefixDetectionStrategy(ProviderStrategy):
    """Strategy for providers that use model prefix detection."""

    def is_provider_model(self, model_entry: Dict[str, Any]) -> bool:
        litellm_params = model_entry.get('litellm_params', {})
        model_id = litellm_params.get('model', '')
        return model_id.startswith(self.config.model_prefix)

    def extract_model_id(self, model_entry: Dict[str, Any]) -> Optional[str]:
        litellm_params = model_entry.get('litellm_params', {})
        model_id = litellm_params.get('model', '')
        if model_id.startswith(self.config.model_prefix):
            return model_id[len(self.config.model_prefix):]
        return None

    def parse_api_model(self, api_model: Dict[str, Any]) -> Dict[str, Any]:
        model_info = {'id': api_model['id'], 'input_cost': None, 'output_cost': None, 'model_info': None}

        default_cost = self.config.pricing.get('default_cost')
        if default_cost is not None:
            model_info['input_cost'] = float(default_cost)
            model_info['output_cost'] = float(default_cost)
            return model_info

        pricing = api_model.get('pricing', {})
        if not isinstance(pricing, dict):
            return model_info

        input_field = self.config.pricing.get('input_field')
        if input_field and input_field in pricing:
            try:
                input_cost = float(pricing[input_field])
                if self.config.pricing.get('is_per_million', False):
                    input_cost = input_cost / self.config.pricing.get('divisor', 1) / 1_000_000
                model_info['input_cost'] = input_cost
            except (ValueError, TypeError):
                pass

        output_field = self.config.pricing.get('output_field')
        if output_field and output_field in pricing:
            try:
                output_cost = float(pricing[output_field])
                if self.config.pricing.get('is_per_million', False):
                    output_cost = output_cost / self.config.pricing.get('divisor', 1) / 1_000_000
                model_info['output_cost'] = output_cost
            except (ValueError, TypeError):
                pass

        if 'model_info' in api_model:
            model_info['model_info'] = api_model['model_info']

        return model_info


class ApiBaseDetectionStrategy(ProviderStrategy):
    """Strategy for providers that use api_base detection."""

    def is_provider_model(self, model_entry: Dict[str, Any]) -> bool:
        litellm_params = model_entry.get('litellm_params', {})
        api_base = str(litellm_params.get('api_base', ''))
        model_id = litellm_params.get('model', '')
        api_base_match = self.config.model_detection['value'] in api_base
        prefix_match = model_id.startswith(self.config.model_prefix)
        return api_base_match and prefix_match

    def extract_model_id(self, model_entry: Dict[str, Any]) -> Optional[str]:
        litellm_params = model_entry.get('litellm_params', {})
        model_id = litellm_params.get('model', '')
        if model_id.startswith(self.config.model_prefix):
            return model_id[len(self.config.model_prefix):]
        return None

    def parse_api_model(self, api_model: Dict[str, Any]) -> Dict[str, Any]:
        model_info = {'id': api_model['id'], 'input_cost': None, 'output_cost': None, 'model_info': None}

        def get_nested_value(data: Dict[str, Any], field_path: str) -> Any:
            keys = field_path.split('.')
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current

        input_field = self.config.pricing.get('input_field')
        if input_field:
            input_value = get_nested_value(api_model, input_field)
            if input_value is not None:
                try:
                    input_cost = float(input_value)
                    if self.config.pricing.get('is_per_million', False):
                        input_cost = input_cost / self.config.pricing.get('divisor', 1) / 1_000_000
                    model_info['input_cost'] = input_cost
                except (ValueError, TypeError):
                    pass

        output_field = self.config.pricing.get('output_field')
        if output_field:
            output_value = get_nested_value(api_model, output_field)
            if output_value is not None:
                try:
                    output_cost = float(output_value)
                    if self.config.pricing.get('is_per_million', False):
                        output_cost = output_cost / self.config.pricing.get('divisor', 1) / 1_000_000
                    model_info['output_cost'] = output_cost
                except (ValueError, TypeError):
                    pass

        if 'model_info' in api_model:
            model_info['model_info'] = api_model['model_info']

        return model_info

    def generate_model_name(self, model_id: str) -> str:
        clean_id = model_id.replace('/', '-').replace(':', '-')
        for cleanup_rule in self.config.model_name_cleanup:
            for replace_old, replace_new in cleanup_rule.get('replace', []):
                clean_id = clean_id.replace(replace_old, replace_new)
        if self.config.model_name_prefix:
            return f"{self.config.model_name_prefix}{clean_id}".lower()
        return clean_id.split('/')[-1].lower()


class ProviderManager:
    """Manages provider configurations and strategies."""

    def __init__(self, config_path: str = "providers.yaml"):
        self.config_path = Path(config_path)
        self.providers: Dict[str, ProviderConfig] = {}
        self.strategies: Dict[str, ProviderStrategy] = {}
        self._api_client = APIClient(timeout=30, max_retries=3, use_cache=True)
        self._load_providers()

    def _load_providers(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)

            if not config_data or 'providers' not in config_data:
                raise ValueError(f"Invalid provider config: missing 'providers' section")

            for provider_name, provider_config in config_data.get('providers', {}).items():
                try:
                    # Handle None values for optional list fields
                    if provider_config.get('model_name_cleanup') is None:
                        provider_config['model_name_cleanup'] = []
                    if provider_config.get('special_models') is None:
                        provider_config['special_models'] = []
                    
                    provider = ProviderConfig(**provider_config)
                    self.providers[provider_name] = provider

                    if provider.model_detection['type'] == 'prefix':
                        strategy = PrefixDetectionStrategy(provider)
                    elif provider.model_detection['type'] == 'api_base':
                        strategy = ApiBaseDetectionStrategy(provider)
                    else:
                        raise ValueError(f"Unknown model detection type: {provider.model_detection['type']}")

                    self.strategies[provider_name] = strategy
                    
                except (TypeError, KeyError) as e:
                    raise ValueError(f"Invalid configuration for provider '{provider_name}': {e}")

        except yaml.YAMLError as e:
            raise ValueError(f"YAML parsing error in provider configuration: {e}")
        except FileNotFoundError:
            raise ValueError(f"Provider configuration file not found: {self.config_path}")
        except Exception as e:
            raise ValueError(f"Error loading provider configuration: {e}")

    def get_provider(self, provider_name: str) -> ProviderConfig:
        if provider_name not in self.providers:
            available = ', '.join(self.providers.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")
        return self.providers[provider_name]

    def get_strategy(self, provider_name: str) -> ProviderStrategy:
        if provider_name not in self.strategies:
            available = ', '.join(self.strategies.keys())
            raise ValueError(f"Unknown provider: {provider_name}. Available: {available}")
        return self.strategies[provider_name]

    def get_api_client(self) -> APIClient:
        """Get the shared API client instance."""
        return self._api_client

    def list_providers(self) -> List[str]:
        return list(self.providers.keys())


class UnifiedModelCleaner:
    """Main class for cleaning up invalid models from LiteLLM config."""

    def __init__(self, config_path: str, provider_names: List[str],
                 dry_run: bool = False, verbose: bool = False):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = setup_logging(verbose, "UnifiedModelCleaner")
        self.provider_manager = ProviderManager()
        self.provider_names = provider_names

        for provider_name in provider_names:
            if provider_name not in self.provider_manager.list_providers():
                raise ValueError(f"Unknown provider: {provider_name}")

    def load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        try:
            self.logger.info(f"Loading configuration from {self.config_path}")
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            if not config:
                raise ValueError("Configuration file is empty or invalid")
            if 'model_list' not in config:
                raise ValueError("Configuration file missing 'model_list' section")

            self.logger.debug(f"Loaded configuration with {len(config['model_list'])} models")
            return config

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the updated configuration back to the file."""
        if self.dry_run:
            self.logger.info("DRY RUN: Would save configuration to file")
            return

        try:
            self.logger.info(f"Saving configuration to {self.config_path}")
            backup_path = self.config_path.with_suffix('.yaml.backup')
            if self.config_path.exists():
                self.config_path.rename(backup_path)
                self.logger.info(f"Created backup at {backup_path}")

            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            self.logger.info("Configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            if backup_path.exists():
                backup_path.rename(self.config_path)
                self.logger.info("Restored configuration from backup")
            raise

    def sort_model_list(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Sort the model list by model_name alphabetically."""
        if 'model_list' not in config or not config['model_list']:
            self.logger.info("No model list found or model list is empty")
            return config, False

        model_list = config['model_list']
        original_order = [(m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', '')) for m in model_list]

        sorted_model_list = sorted(
            model_list,
            key=lambda x: (x.get('model_name', 'unnamed').lower(), x.get('litellm_params', {}).get('model', '').lower())
        )

        sorted_order = [(m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', '')) for m in sorted_model_list]
        was_sorted = original_order != sorted_order

        if was_sorted:
            config['model_list'] = sorted_model_list
            self.logger.info(f"Sorted {len(model_list)} models by model_name")
        else:
            self.logger.info("Model list is already sorted")

        return config, was_sorted

    def extract_provider_models(self, config: Dict[str, Any], provider_name: str) -> List[Tuple[int, str, str]]:
        """Extract models for a specific provider from the configuration."""
        strategy = self.provider_manager.get_strategy(provider_name)
        provider_models = []
        model_list = config.get('model_list', [])

        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
            if strategy.is_provider_model(model_entry):
                model_id = strategy.extract_model_id(model_entry)
                model_name = model_entry.get('model_name', 'unnamed')
                full_model_id = model_entry.get('litellm_params', {}).get('model', '')
                if model_id:
                    provider_models.append((index, full_model_id, model_name))
                    self.logger.debug(f"Found {provider_name} model: {full_model_id} (name: {model_name})")

        self.logger.info(f"Found {len(provider_models)} {provider_name} models in configuration")
        return provider_models

    def fetch_available_models(self, provider_name: str) -> Dict[str, Dict[str, Any]]:
        """Fetch available models with pricing from provider API."""
        provider = self.provider_manager.get_provider(provider_name)
        strategy = self.provider_manager.get_strategy(provider_name)
        api_client = self.provider_manager.get_api_client()

        try:
            self.logger.info(f"Fetching available models from {provider.name} API...")
            headers = {}
            if provider.api_key_env:
                api_key = os.environ.get(provider.api_key_env)
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'

            data = api_client.fetch(provider.api_url, headers=headers or None, logger=self.logger)

            if 'data' not in data:
                raise ValueError("Invalid API response format: missing 'data' field")

            available_models = {}
            for model in data['data']:
                if isinstance(model, dict) and 'id' in model:
                    model_info = strategy.parse_api_model(model)
                    available_models[model_info['id']] = model_info

            # Fetch embedding models if configured
            if provider.embeddings_api_url:
                try:
                    self.logger.debug(f"Fetching embedding models from {provider.name}...")
                    embed_data = api_client.fetch(
                        provider.embeddings_api_url, 
                        headers=headers or None, 
                        logger=self.logger
                    )
                    if 'data' in embed_data:
                        for model in embed_data['data']:
                            if isinstance(model, dict) and 'id' in model:
                                model_with_info = dict(model)
                                if 'model_info' not in model_with_info:
                                    model_with_info['model_info'] = {'mode': 'embedding'}
                                model_info = strategy.parse_api_model(model_with_info)
                                available_models[model_info['id']] = model_info
                        self.logger.debug(f"Fetched embedding models from {provider.name}")
                except requests.RequestException as e:
                    self.logger.warning(f"Could not fetch embedding models: {e}")

            self.logger.info(f"Fetched {len(available_models)} available models from {provider.name} API")
            return available_models

        except requests.RequestException as e:
            self.logger.error(f"API request failed for {provider.name}: {e}")
            raise


    def validate_models(self, config_models: List[Tuple[int, str, str]],
                       api_models: Dict[str, Dict[str, Any]],
                       provider_name: str) -> List[Tuple[int, str, str]]:
        """Compare config models with API models and identify invalid ones."""
        provider = self.provider_manager.get_provider(provider_name)
        strategy = self.provider_manager.get_strategy(provider_name)
        api_models_set = set(api_models.keys())
        invalid_models = []

        for index, full_model_id, model_name in config_models:
            model_entry = {'litellm_params': {'model': full_model_id}}
            api_model_id = strategy.extract_model_id(model_entry)

            if not api_model_id:
                invalid_models.append((index, full_model_id, model_name))
                continue

            if api_model_id in provider.special_models:
                self.logger.debug(f"Special model (will not be removed): {full_model_id}")
                continue

            if api_model_id not in api_models_set:
                invalid_models.append((index, full_model_id, model_name))
                self.logger.debug(f"Invalid model found: {full_model_id}")
            else:
                self.logger.debug(f"Valid model: {full_model_id}")

        self.logger.info(f"Identified {len(invalid_models)} invalid {provider.name} models")
        return invalid_models

    def validate_and_update_costs(self, config: Dict[str, Any],
                                 config_models: List[Tuple[int, str, str]],
                                 api_models: Dict[str, Dict[str, Any]],
                                 provider_name: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Validate and update model costs based on API pricing."""
        provider = self.provider_manager.get_provider(provider_name)
        strategy = self.provider_manager.get_strategy(provider_name)
        cost_changes = []
        model_list = config['model_list']

        for index, full_model_id, model_name in config_models:
            model_entry = {'litellm_params': {'model': full_model_id}}
            api_model_id = strategy.extract_model_id(model_entry)

            if not api_model_id or api_model_id not in api_models:
                continue

            api_model_info = api_models[api_model_id]
            api_input_cost = api_model_info.get('input_cost')
            api_output_cost = api_model_info.get('output_cost')

            if 0 <= index < len(model_list):
                model_entry = model_list[index]
                litellm_params = model_entry.get('litellm_params', {})
                current_input_cost = litellm_params.get('input_cost_per_token')
                current_output_cost = litellm_params.get('output_cost_per_token')

                # Convert string costs to float
                if current_input_cost is not None and isinstance(current_input_cost, str):
                    try:
                        current_input_cost = float(current_input_cost)
                    except (ValueError, TypeError):
                        current_input_cost = None
                if current_output_cost is not None and isinstance(current_output_cost, str):
                    try:
                        current_output_cost = float(current_output_cost)
                    except (ValueError, TypeError):
                        current_output_cost = None

                input_changed = False
                output_changed = False
                change_info = {'index': index, 'model_id': full_model_id, 'model_name': model_name, 'changes': {}}

                if api_input_cost is not None:
                    adjusted_input_cost = adjust_cost_for_free_model(api_input_cost) if provider.pricing.get('free_model_handling', False) else api_input_cost
                    if current_input_cost is None or not costs_are_equal(current_input_cost, adjusted_input_cost):
                        input_changed = True
                        change_info['changes']['input_cost'] = {'old': current_input_cost, 'new': adjusted_input_cost}
                        litellm_params['input_cost_per_token'] = adjusted_input_cost

                if api_output_cost is not None:
                    adjusted_output_cost = adjust_cost_for_free_model(api_output_cost) if provider.pricing.get('free_model_handling', False) else api_output_cost
                    if current_output_cost is None or not costs_are_equal(current_output_cost, adjusted_output_cost):
                        output_changed = True
                        change_info['changes']['output_cost'] = {'old': current_output_cost, 'new': adjusted_output_cost}
                        litellm_params['output_cost_per_token'] = adjusted_output_cost

                if input_changed or output_changed:
                    cost_changes.append(change_info)
                    self._log_cost_change(full_model_id, model_name, change_info, current_input_cost, current_output_cost)

        if cost_changes:
            self.logger.info(f"Identified {len(cost_changes)} models with cost updates")
        else:
            self.logger.info("No cost updates needed - all costs are current")

        return config, cost_changes

    def _log_cost_change(self, model_id: str, model_name: str, change_info: Dict[str, Any],
                        current_input_cost: Optional[float], current_output_cost: Optional[float]) -> None:
        """Log cost change details."""
        self.logger.info(f"Cost update for {model_id} (name: {model_name})")
        if 'input_cost' in change_info['changes']:
            old_val = current_input_cost if current_input_cost is not None else "None"
            new_val = change_info['changes']['input_cost']['new']
            pct_str = ""
            if current_input_cost is not None and current_input_cost != 0:
                pct_change = ((new_val - current_input_cost) / current_input_cost) * 100
                pct_str = f" ({pct_change:+.1f}%)"
            self.logger.info(f"  Input cost: {old_val} → {new_val}{pct_str}")

        if 'output_cost' in change_info['changes']:
            old_val = current_output_cost if current_output_cost is not None else "None"
            new_val = change_info['changes']['output_cost']['new']
            pct_str = ""
            if current_output_cost is not None and current_output_cost != 0:
                pct_change = ((new_val - current_output_cost) / current_output_cost) * 100
                pct_str = f" ({pct_change:+.1f}%)"
            self.logger.info(f"  Output cost: {old_val} → {new_val}{pct_str}")

    def remove_invalid_entries(self, config: Dict[str, Any],
                              invalid_models: List[Tuple[int, str, str]]) -> Dict[str, Any]:
        """Remove invalid model entries from the configuration."""
        if not invalid_models:
            self.logger.info("No invalid models to remove")
            return config

        invalid_indices = sorted([index for index, _, _ in invalid_models], reverse=True)
        model_list = config['model_list']
        removed_count = 0

        for index in invalid_indices:
            if 0 <= index < len(model_list):
                removed_model = model_list.pop(index)
                removed_count += 1
                model_name = removed_model.get('model_name', 'unnamed')
                model_id = removed_model.get('litellm_params', {}).get('model', 'unknown')
                self.logger.info(f"Removed model entry: {model_id} (name: {model_name})")

        self.logger.info(f"Successfully removed {removed_count} invalid model entries")
        return config

    def preview_changes(self, invalid_models: List[Tuple[int, str, str]], provider_name: str) -> None:
        """Preview what changes would be made in dry-run mode."""
        provider = self.provider_manager.get_provider(provider_name)
        if not invalid_models:
            self.logger.info(f"[DRY-RUN] No invalid {provider.name} models found. No changes needed.")
            return

        self.logger.info(f"[DRY-RUN] Would remove the following invalid {provider.name} models:")
        for _, model_id, model_name in invalid_models:
            self.logger.info(f"  - Model: {model_id} (model_name: {model_name})")
        self.logger.info(f"[DRY-RUN] {len(invalid_models)} model entries would be removed")
        self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")


    def add_model_to_config(self, config: Dict[str, Any], model_ids: List[str],
                           provider_name: str, available_models: Dict[str, Dict[str, Any]],
                           custom_model_name: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Add one or more models to the configuration for a specific provider."""
        strategy = self.provider_manager.get_strategy(provider_name)
        provider = self.provider_manager.get_provider(provider_name)
        added_models = []
        failed_models = []

        existing_models = self.extract_provider_models(config, provider_name)
        existing_model_ids = []
        for _, full_model_id, _ in existing_models:
            model_entry = {'litellm_params': {'model': full_model_id}}
            model_id = strategy.extract_model_id(model_entry)
            if model_id:
                existing_model_ids.append(model_id)
        existing_names = [name for _, _, name in existing_models]

        for model_id in model_ids:
            self.logger.info(f"Processing model: {model_id}")

            if model_id not in available_models:
                self.logger.error(f"Model '{model_id}' not found in {provider.name} API")
                failed_models.append(model_id)
                continue

            if model_id in existing_model_ids:
                self.logger.warning(f"Model '{model_id}' already exists in configuration")
                failed_models.append(model_id)
                continue

            model_entry = strategy.create_model_entry(model_id, available_models[model_id])

            if custom_model_name and len(model_ids) == 1:
                model_entry['model_name'] = custom_model_name

            model_name = model_entry['model_name']
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1
            model_entry['model_name'] = model_name

            if not self.dry_run:
                config['model_list'].append(model_entry)

            full_model_id = model_entry['litellm_params']['model']
            added_models.append(full_model_id)
            existing_model_ids.append(model_id)
            existing_names.append(model_name)

            self.logger.info(f"Added model '{full_model_id}' with name '{model_name}'")
            input_cost = model_entry['litellm_params'].get('input_cost_per_token')
            output_cost = model_entry['litellm_params'].get('output_cost_per_token')
            if input_cost is not None:
                self.logger.info(f"  Input cost: {input_cost}")
            if output_cost is not None:
                self.logger.info(f"  Output cost: {output_cost}")

            # Special handling for OpenRouter: check for free variant
            if provider_name == 'openrouter':
                free_variant_id = f"{model_id}:free"
                if free_variant_id in available_models:
                    if free_variant_id in existing_model_ids:
                        self.logger.warning(f"Free variant '{free_variant_id}' already exists in configuration")
                    else:
                        free_model_entry = strategy.create_model_entry(free_variant_id, available_models[free_variant_id])
                        # Use the same model name as the base model
                        free_model_entry['model_name'] = model_name
                        
                        if not self.dry_run:
                            config['model_list'].append(free_model_entry)
                        
                        free_full_model_id = free_model_entry['litellm_params']['model']
                        added_models.append(free_full_model_id)
                        existing_model_ids.append(free_variant_id)
                        
                        self.logger.info(f"Added free variant '{free_full_model_id}' with name '{model_name}'")
                        free_input_cost = free_model_entry['litellm_params'].get('input_cost_per_token')
                        free_output_cost = free_model_entry['litellm_params'].get('output_cost_per_token')
                        if free_input_cost is not None:
                            self.logger.info(f"  Input cost: {free_input_cost}")
                        if free_output_cost is not None:
                            self.logger.info(f"  Output cost: {free_output_cost}")

        if failed_models:
            self.logger.warning(f"Failed to add {len(failed_models)} model(s): {', '.join(failed_models)}")
        if added_models:
            self.logger.info(f"Successfully processed {len(added_models)} model(s) for addition")

        return config, added_models

    def preview_add_model(self, model_ids: List[str], provider_name: str,
                         available_models: Dict[str, Dict[str, Any]],
                         custom_model_name: Optional[str] = None) -> None:
        """Preview what models would be added in dry-run mode."""
        strategy = self.provider_manager.get_strategy(provider_name)
        provider = self.provider_manager.get_provider(provider_name)
        existing_models = self.extract_provider_models(self.load_config(), provider_name)

        existing_model_ids = []
        for _, full_model_id, _ in existing_models:
            model_entry = {'litellm_params': {'model': full_model_id}}
            model_id = strategy.extract_model_id(model_entry)
            if model_id:
                existing_model_ids.append(model_id)
        existing_names = [name for _, _, name in existing_models]

        previewed_additions = []
        for model_id in model_ids:
            if model_id not in available_models:
                self.logger.error(f"[DRY-RUN] Model '{model_id}' not found in {provider_name} API")
                continue
            if model_id in existing_model_ids:
                self.logger.warning(f"[DRY-RUN] Model '{model_id}' already exists in configuration")
                continue

            api_model_info = available_models[model_id]
            model_entry = strategy.create_model_entry(model_id, api_model_info)

            if custom_model_name and len(model_ids) == 1:
                model_entry['model_name'] = custom_model_name

            model_name = model_entry['model_name']
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1
            model_entry['model_name'] = model_name

            self.logger.info(f"[DRY-RUN] Would add model '{model_entry['litellm_params']['model']}' with name '{model_entry['model_name']}'")
            input_cost = model_entry['litellm_params'].get('input_cost_per_token')
            output_cost = model_entry['litellm_params'].get('output_cost_per_token')
            if input_cost is not None and output_cost is not None:
                self.logger.info(f"[DRY-RUN]   Input cost: {input_cost}, Output cost: {output_cost}")

            # Special handling for OpenRouter: check for free variant
            if provider_name == 'openrouter':
                free_variant_id = f"{model_id}:free"
                if free_variant_id in available_models:
                    if free_variant_id in existing_model_ids:
                        self.logger.info(f"[DRY-RUN]   Free variant '{free_variant_id}' already exists in configuration")
                    else:
                        free_model_entry = strategy.create_model_entry(free_variant_id, available_models[free_variant_id])
                        # Use the same model name as the base model
                        free_model_entry['model_name'] = model_name
                        self.logger.info(f"[DRY-RUN]   Free variant '{free_model_entry['litellm_params']['model']}' would also be added with name '{model_name}'")
                        free_input_cost = free_model_entry['litellm_params'].get('input_cost_per_token')
                        free_output_cost = free_model_entry['litellm_params'].get('output_cost_per_token')
                        if free_input_cost is not None and free_output_cost is not None:
                            self.logger.info(f"[DRY-RUN]     Input cost: {free_input_cost}, Output cost: {free_output_cost}")

            previewed_additions.append(model_id)
            existing_model_ids.append(model_id)
            existing_names.append(model_name)

        if previewed_additions:
            self.logger.info(f"[DRY-RUN] Would add {len(previewed_additions)} model(s)")
        else:
            self.logger.info("[DRY-RUN] No valid models to add.")
        self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")

    def cleanup_provider(self, provider_name: str, add_models: Optional[List[str]] = None,
                        custom_model_name: Optional[str] = None) -> Tuple[int, int, List[str]]:
        """Clean up models for a specific provider."""
        try:
            config = self.load_config()

            if add_models:
                available_models = self.fetch_available_models(provider_name)
                if self.dry_run:
                    self.preview_add_model(add_models, provider_name, available_models, custom_model_name)
                    return 0, 0, [f"Previewed {len(add_models)} models for addition"]
                else:
                    config, added_models = self.add_model_to_config(config, add_models, provider_name, available_models, custom_model_name)
                    if added_models:
                        config, was_sorted = self.sort_model_list(config)
                        self.save_config(config)
                        self.logger.info(f"✅ Successfully added {len(added_models)} model(s)")
                    else:
                        self.logger.warning("⚠️ No models were added")
                return 0, 0, [f"Added {len(added_models)} model(s)"]

            config, was_sorted = self.sort_model_list(config)
            available_models = self.fetch_available_models(provider_name)
            provider_models = self.extract_provider_models(config, provider_name)

            if not provider_models:
                if self.dry_run:
                    self.logger.info(f"[DRY-RUN] No {provider_name} models found to process")
                else:
                    if was_sorted:
                        self.save_config(config)
                    self.logger.info(f"No {provider_name} models found to process")
                return 0, 0, []

            invalid_models = self.validate_models(provider_models, available_models, provider_name)
            config, cost_changes = self.validate_and_update_costs(config, provider_models, available_models, provider_name)

            if self.dry_run:
                self.preview_changes(invalid_models, provider_name)
            else:
                if invalid_models:
                    config = self.remove_invalid_entries(config, invalid_models)
                if was_sorted or invalid_models or cost_changes:
                    self.save_config(config)

            models_removed = len(invalid_models)
            models_updated = len(cost_changes)
            changes_made = []
            if models_removed > 0:
                changes_made.append(f"Removed {models_removed} invalid models")
            if models_updated > 0:
                changes_made.append(f"Updated costs for {models_updated} models")
            if was_sorted:
                changes_made.append("Sorted model list")

            return models_removed, models_updated, changes_made

        except Exception as e:
            self.logger.error(f"Cleanup failed for {provider_name}: {e}")
            raise

    def cleanup_all_providers(self, add_models: Optional[Dict[str, List[str]]] = None,
                             custom_model_name: Optional[str] = None) -> Dict[str, Tuple[int, int, List[str]]]:
        """Clean up models for all configured providers."""
        results = {}
        for provider_name in self.provider_names:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing provider: {provider_name}")
            self.logger.info(f"{'='*60}")

            provider_add_models = add_models.get(provider_name) if add_models else None
            try:
                models_removed, models_updated, changes_made = self.cleanup_provider(provider_name, provider_add_models, custom_model_name)
                results[provider_name] = (models_removed, models_updated, changes_made)
            except Exception as e:
                self.logger.error(f"Error processing provider {provider_name}: {e}")
                results[provider_name] = (0, 0, [f"Error: {str(e)}"])

        return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean up invalid models, update costs, sort model list, and add new models in LiteLLM configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs four main functions:
1. Sorts the model list alphabetically by model_name
2. Validates models against the current API and removes invalid entries
3. Updates model costs when they differ from API pricing
4. Adds one or more models to the configuration

Supported providers: openrouter, requesty, nano_gpt, vercel, poe, nvidia, synthetic, all

Examples:
  %(prog)s --provider openrouter                           # Process OpenRouter models
  %(prog)s --provider all                                  # Process all providers
  %(prog)s --provider requesty --add-model "model1 model2" # Add models to Requesty
  %(prog)s --provider openrouter --dry-run --verbose       # Preview changes
        """
    )

    parser.add_argument('--provider', '-p', required=True,
                       help='Provider to process (openrouter, requesty, nano_gpt, vercel, poe, nvidia, or "all")')
    parser.add_argument('--config', '-c', default="config.yaml",
                       help='Path to LiteLLM configuration file (default: config.yaml)')
    parser.add_argument('--dry-run', '-d', action='store_true',
                       help='Preview all changes without modifying the configuration file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging output')
    parser.add_argument('--add-model', nargs='+',
                       help='Add one or more models to the configuration')
    parser.add_argument('--model-name',
                       help='Custom model name (only valid when adding a single model)')

    args = parser.parse_args()

    try:
        load_dotenv(override=True)

        temp_manager = ProviderManager()
        available_providers = temp_manager.list_providers()

        if args.provider.lower() == 'all':
            provider_names = available_providers
        else:
            provider_names = [args.provider.lower()]

        for provider_name in provider_names:
            if provider_name not in available_providers:
                print(f"Error: Unknown provider '{provider_name}'. Available: {', '.join(available_providers)}", file=sys.stderr)
                return 1

        add_models = None
        if args.add_model:
            add_models = {}
            for model_spec in args.add_model:
                if '|' in model_spec:
                    provider, model_id = model_spec.split('|', 1)
                    if provider not in provider_names:
                        print(f"Warning: Provider '{provider}' not in selected providers, skipping '{model_id}'", file=sys.stderr)
                        continue
                else:
                    if len(provider_names) == 1:
                        provider = provider_names[0]
                        model_id = model_spec
                    else:
                        print(f"Error: Must specify provider for model '{model_spec}' when using multiple providers. Use format provider|model", file=sys.stderr)
                        return 1

                if provider not in add_models:
                    add_models[provider] = []
                add_models[provider].append(model_id)

        cleaner = UnifiedModelCleaner(
            config_path=args.config,
            provider_names=provider_names,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        if len(provider_names) == 1:
            provider_add_models = add_models.get(provider_names[0]) if add_models else None
            models_removed, models_updated, changes_made = cleaner.cleanup_provider(provider_names[0], provider_add_models, args.model_name)

            print(f"\nSummary for {provider_names[0]}:")
            print(f"Models removed: {models_removed}")
            print(f"Models updated: {models_updated}")
            print(f"Total changes: {len(changes_made)}")

            if changes_made:
                print(f"\nChanges made:")
                for change in changes_made:
                    print(f"  - {change}")
        else:
            results = cleaner.cleanup_all_providers(add_models, args.model_name)

            print(f"\nSummary for all providers:")
            total_removed = 0
            total_updated = 0
            total_changes = 0

            for provider_name, (removed, updated, changes) in results.items():
                print(f"\n{provider_name}:")
                print(f"  Models removed: {removed}")
                print(f"  Models updated: {updated}")
                print(f"  Changes: {len(changes)}")

                total_removed += removed
                total_updated += updated
                total_changes += len(changes)

                if changes:
                    print(f"  Details:")
                    for change in changes:
                        print(f"    - {change}")

            print(f"\nOverall Summary:")
            print(f"Total models removed: {total_removed}")
            print(f"Total models updated: {total_updated}")
            print(f"Total changes: {total_changes}")

        if args.dry_run:
            print(f"\nDRY RUN: No actual changes were made")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
