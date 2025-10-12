#!/usr/bin/env python3
"""
Unified Model Cleanup and Cost Update Script for LiteLLM Config

This script validates models from multiple providers in a LiteLLM config.yaml file
against their current APIs and:
1. Removes any invalid model entries (except special models)
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing
3. Sorts the model list alphabetically
4. Adds new models when requested

Supported providers: openrouter, requesty, novita, nano_gpt, all

API Key Support:
    Providers that require API keys for model listing:
    - Requesty: Set REQUESTY_API_KEY environment variable
    - Nano-GPT: Set NANOGPT_API_KEY environment variable
    - OpenRouter: No API key required for model listing

Usage:
    python cleanup_models.py --provider openrouter [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_models.py --provider all [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_models.py --provider requesty --add-model "model1 model2"

Author: Unified script for LiteLLM Config Management
"""

import argparse
import logging
import os
import sys
import yaml
import requests
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from dotenv import load_dotenv


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
    model_name_cleanup: List[Dict[str, str]]
    special_models: List[str]
    api_base_config: Optional[Dict[str, str]] = None
    api_key_env: Optional[str] = None


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

    @abstractmethod
    def generate_model_name(self, model_id: str) -> str:
        """Generate a model name for this provider."""
        pass

    @abstractmethod
    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new model entry for the config."""
        pass


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
        model_info = {
            'id': api_model['id'],
            'input_cost': None,
            'output_cost': None
        }

        pricing = api_model.get('pricing', {})
        if not isinstance(pricing, dict):
            return model_info

        # Extract input cost
        input_field = self.config.pricing['input_field']
        if input_field in pricing:
            try:
                input_cost = float(pricing[input_field])
                if self.config.pricing.get('is_per_million', False):
                    divisor = self.config.pricing.get('divisor', 1)
                    input_cost = input_cost / divisor / 1_000_000
                model_info['input_cost'] = input_cost
            except (ValueError, TypeError):
                pass

        # Extract output cost
        output_field = self.config.pricing['output_field']
        if output_field in pricing:
            try:
                output_cost = float(pricing[output_field])
                if self.config.pricing.get('is_per_million', False):
                    divisor = self.config.pricing.get('divisor', 1)
                    output_cost = output_cost / divisor / 1_000_000
                model_info['output_cost'] = output_cost
            except (ValueError, TypeError):
                pass

        return model_info

    def generate_model_name(self, model_id: str) -> str:
        # Clean up the model ID
        clean_id = model_id.replace('/', '-').replace(':', '-')

        # Apply name cleanup rules
        for cleanup_rule in self.config.model_name_cleanup:
            for replace_old, replace_new in cleanup_rule.get('replace', []):
                clean_id = clean_id.replace(replace_old, replace_new)

        # Add provider prefix
        model_name = f"{self.config.model_name_prefix}{clean_id}"
        return model_name

    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any]) -> Dict[str, Any]:
        model_name = self.generate_model_name(model_id)

        # Handle free models
        input_cost = api_model_info.get('input_cost')
        output_cost = api_model_info.get('output_cost')

        if self.config.pricing.get('free_model_handling', False):
            if input_cost is not None and input_cost == 0.0:
                input_cost = 1.0e-09
            if output_cost is not None and output_cost == 0.0:
                output_cost = 1.0e-09

        model_entry = {
            'model_name': model_name,
            'litellm_params': {
                'model': f"{self.config.model_prefix}{model_id}"
            }
        }

        # Add costs if available
        if input_cost is not None:
            model_entry['litellm_params']['input_cost_per_token'] = input_cost
        if output_cost is not None:
            model_entry['litellm_params']['output_cost_per_token'] = output_cost

        return model_entry


class ApiBaseDetectionStrategy(ProviderStrategy):
    """Strategy for providers that use api_base detection."""

    def is_provider_model(self, model_entry: Dict[str, Any]) -> bool:
        litellm_params = model_entry.get('litellm_params', {})
        api_base = str(litellm_params.get('api_base', ''))
        model_id = litellm_params.get('model', '')

        # Check both api_base pattern and model prefix
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
        model_info = {
            'id': api_model['id'],
            'input_cost': None,
            'output_cost': None
        }

        def get_nested_value(data: Dict[str, Any], field_path: str) -> Any:
            """Get value from nested dict using dot notation path."""
            keys = field_path.split('.')
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current

        # Extract input cost
        input_field = self.config.pricing['input_field']
        input_value = get_nested_value(api_model, input_field)
        if input_value is not None:
            try:
                input_cost = float(input_value)
                if self.config.pricing.get('is_per_million', False):
                    divisor = self.config.pricing.get('divisor', 1)
                    input_cost = input_cost / divisor / 1_000_000
                model_info['input_cost'] = input_cost
            except (ValueError, TypeError):
                pass

        # Extract output cost
        output_field = self.config.pricing['output_field']
        output_value = get_nested_value(api_model, output_field)
        if output_value is not None:
            try:
                output_cost = float(output_value)
                if self.config.pricing.get('is_per_million', False):
                    divisor = self.config.pricing.get('divisor', 1)
                    output_cost = output_cost / divisor / 1_000_000
                model_info['output_cost'] = output_cost
            except (ValueError, TypeError):
                pass

        return model_info

    def generate_model_name(self, model_id: str) -> str:
        # For API base detection, often use base name directly
        clean_id = model_id.replace('/', '-').replace(':', '-')

        # Apply name cleanup rules
        for cleanup_rule in self.config.model_name_cleanup:
            for replace_old, replace_new in cleanup_rule.get('replace', []):
                clean_id = clean_id.replace(replace_old, replace_new)

        # Add provider prefix if specified
        if self.config.model_name_prefix:
            model_name = f"{self.config.model_name_prefix}{clean_id}"
        else:
            # Use base name for load balancing compatibility
            model_name = clean_id.split('/')[-1]  # Extract base name

        return model_name

    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any]) -> Dict[str, Any]:
        model_name = self.generate_model_name(model_id)

        # Handle free models
        input_cost = api_model_info.get('input_cost')
        output_cost = api_model_info.get('output_cost')

        if self.config.pricing.get('free_model_handling', False):
            if input_cost is not None and input_cost == 0.0:
                input_cost = 1.0e-09
            if output_cost is not None and output_cost == 0.0:
                output_cost = 1.0e-09

        model_entry = {
            'model_name': model_name,
            'litellm_params': {
                'model': f"{self.config.model_prefix}{model_id}"
            }
        }

        # Add API base configuration if specified
        if self.config.api_base_config:
            if 'url' in self.config.api_base_config:
                model_entry['litellm_params']['api_base'] = self.config.api_base_config['url']
            elif 'url_env' in self.config.api_base_config:
                model_entry['litellm_params']['api_base'] = f"os.environ/{self.config.api_base_config['url_env']}"

            if 'api_key_env' in self.config.api_base_config:
                model_entry['litellm_params']['api_key'] = f"os.environ/{self.config.api_base_config['api_key_env']}"

        # Add costs if available
        if input_cost is not None:
            model_entry['litellm_params']['input_cost_per_token'] = input_cost
        if output_cost is not None:
            model_entry['litellm_params']['output_cost_per_token'] = output_cost

        return model_entry


class ProviderManager:
    """Manages provider configurations and strategies."""

    def __init__(self, config_path: str = "providers.yaml"):
        self.config_path = Path(config_path)
        self.providers: Dict[str, ProviderConfig] = {}
        self.strategies: Dict[str, ProviderStrategy] = {}
        self._load_providers()

    def _load_providers(self):
        """Load provider configurations from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)

            for provider_name, provider_config in config_data.get('providers', {}).items():
                provider = ProviderConfig(**provider_config)
                self.providers[provider_name] = provider

                # Create appropriate strategy
                if provider.model_detection['type'] == 'prefix':
                    strategy = PrefixDetectionStrategy(provider)
                elif provider.model_detection['type'] == 'api_base':
                    strategy = ApiBaseDetectionStrategy(provider)
                else:
                    raise ValueError(f"Unknown model detection type: {provider.model_detection['type']}")

                self.strategies[provider_name] = strategy

        except Exception as e:
            raise ValueError(f"Error loading provider configuration: {e}")

    def get_provider(self, provider_name: str) -> ProviderConfig:
        """Get provider configuration by name."""
        if provider_name not in self.providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.providers[provider_name]

    def get_strategy(self, provider_name: str) -> ProviderStrategy:
        """Get provider strategy by name."""
        if provider_name not in self.strategies:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self.strategies[provider_name]

    def list_providers(self) -> List[str]:
        """List all available providers."""
        return list(self.providers.keys())


class UnifiedModelCleaner:
    """Main class for cleaning up invalid models from LiteLLM config."""

    def __init__(self, config_path: str, provider_names: List[str],
                 dry_run: bool = False, verbose: bool = False):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.provider_manager = ProviderManager()
        self.provider_names = provider_names

        # Validate provider names
        for provider_name in provider_names:
            if provider_name not in self.provider_manager.list_providers():
                raise ValueError(f"Unknown provider: {provider_name}")

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)

        return logger

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
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

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

        try:
            self.logger.info(f"Fetching available models with pricing from {provider.name} API...")

            # Prepare headers
            headers = {}
            if provider.api_key_env:
                api_key = os.environ.get(provider.api_key_env)
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
                    self.logger.debug(f"Using API key from environment variable: {provider.api_key_env}")
                else:
                    self.logger.warning(f"API key environment variable '{provider.api_key_env}' not found, proceeding without authentication")

            response = requests.get(provider.api_url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data:
                raise ValueError("Invalid API response format: missing 'data' field")

            # Extract model IDs and pricing from the API response
            available_models = {}
            for model in data['data']:
                if isinstance(model, dict) and 'id' in model:
                    model_info = strategy.parse_api_model(model)
                    available_models[model_info['id']] = model_info

            self.logger.info(f"Fetched {len(available_models)} available models from {provider.name} API")
            return available_models

        except requests.RequestException as e:
            self.logger.error(f"API request failed for {provider.name}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing API response for {provider.name}: {e}")
            raise

    def validate_models(self, config_models: List[Tuple[int, str, str]],
                       api_models: Dict[str, Dict[str, Any]],
                       provider_name: str) -> List[Tuple[int, str, str]]:
        """Compare config models with API models and identify invalid ones."""
        provider = self.provider_manager.get_provider(provider_name)
        strategy = self.provider_manager.get_strategy(provider_name)

        # Convert API models to set for faster lookup
        api_models_set = set(api_models.keys())
        invalid_models = []

        for index, full_model_id, model_name in config_models:
            # Extract the model ID using provider-specific logic
            model_entry = {'litellm_params': {'model': full_model_id}}
            api_model_id = strategy.extract_model_id(model_entry)

            if not api_model_id:
                invalid_models.append((index, full_model_id, model_name))
                continue

            # Check if this is a special model that should never be removed
            if api_model_id in provider.special_models:
                self.logger.debug(f"Special model found (will not be removed): {full_model_id} -> {api_model_id}")
                continue

            if api_model_id not in api_models_set:
                invalid_models.append((index, full_model_id, model_name))
                self.logger.debug(f"Invalid model found: {full_model_id} -> {api_model_id}")
            else:
                self.logger.debug(f"Valid model: {full_model_id} -> {api_model_id}")

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
            # Extract the API model ID
            model_entry = {'litellm_params': {'model': full_model_id}}
            api_model_id = strategy.extract_model_id(model_entry)

            if not api_model_id or api_model_id not in api_models:
                continue

            api_model_info = api_models[api_model_id]
            api_input_cost = api_model_info.get('input_cost')
            api_output_cost = api_model_info.get('output_cost')

            # Get current costs from config
            if 0 <= index < len(model_list):
                model_entry = model_list[index]
                litellm_params = model_entry.get('litellm_params', {})
                current_input_cost = litellm_params.get('input_cost_per_token')
                current_output_cost = litellm_params.get('output_cost_per_token')

                # Convert string costs to float if necessary
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

                # Check for cost changes
                input_changed = False
                output_changed = False
                change_info = {
                    'index': index,
                    'model_id': full_model_id,
                    'model_name': model_name,
                    'changes': {}
                }

                # Compare input costs
                if api_input_cost is not None:
                    # Handle free models: if API returns 0.0, use 1e-09 for LiteLLM compatibility
                    adjusted_input_cost = 1e-09 if api_input_cost == 0.0 and provider.pricing.get('free_model_handling', False) else api_input_cost

                    # Compare costs with 2 decimal places precision to avoid floating-point issues
                    if current_input_cost is not None:
                        current_rounded = round(current_input_cost, 2)
                        adjusted_rounded = round(adjusted_input_cost, 2)
                        if current_rounded != adjusted_rounded:
                            input_changed = True
                            change_info['changes']['input_cost'] = {
                                'old': current_input_cost,
                                'new': adjusted_input_cost
                            }
                            litellm_params['input_cost_per_token'] = adjusted_input_cost
                            self.logger.debug(f"Input cost change for {full_model_id}: {current_input_cost} → {adjusted_input_cost}")
                    else:
                        # Current cost is None, always update
                        input_changed = True
                        change_info['changes']['input_cost'] = {
                            'old': current_input_cost,
                            'new': adjusted_input_cost
                        }
                        litellm_params['input_cost_per_token'] = adjusted_input_cost
                        self.logger.debug(f"Input cost change for {full_model_id}: {current_input_cost} → {adjusted_input_cost}")

                # Compare output costs
                if api_output_cost is not None:
                    # Handle free models: if API returns 0.0, use 1e-09 for LiteLLM compatibility
                    adjusted_output_cost = 1e-09 if api_output_cost == 0.0 and provider.pricing.get('free_model_handling', False) else api_output_cost

                    # Compare costs with 2 decimal places precision to avoid floating-point issues
                    if current_output_cost is not None:
                        current_rounded = round(current_output_cost, 2)
                        adjusted_rounded = round(adjusted_output_cost, 2)
                        if current_rounded != adjusted_rounded:
                            output_changed = True
                            change_info['changes']['output_cost'] = {
                                'old': current_output_cost,
                                'new': adjusted_output_cost
                            }
                            litellm_params['output_cost_per_token'] = adjusted_output_cost
                            self.logger.debug(f"Output cost change for {full_model_id}: {current_output_cost} → {adjusted_output_cost}")
                    else:
                        # Current cost is None, always update
                        output_changed = True
                        change_info['changes']['output_cost'] = {
                            'old': current_output_cost,
                            'new': adjusted_output_cost
                        }
                        litellm_params['output_cost_per_token'] = adjusted_output_cost
                        self.logger.debug(f"Output cost change for {full_model_id}: {current_output_cost} → {adjusted_output_cost}")

                # Record changes if any occurred
                if input_changed or output_changed:
                    cost_changes.append(change_info)
                    self.logger.info(f"Cost update for {full_model_id} (name: {model_name})")

                    if input_changed:
                        old_val = current_input_cost if current_input_cost is not None else "None"
                        new_val = change_info['changes']['input_cost']['new']

                        # Calculate percentage change
                        if current_input_cost is not None and current_input_cost != 0:
                            pct_change = ((new_val - current_input_cost) / current_input_cost) * 100
                            pct_str = f" ({pct_change:+.1f}%)"
                        else:
                            pct_str = ""

                        self.logger.info(f"  Input cost: {old_val} → {new_val}{pct_str}")

                    if output_changed:
                        old_val = current_output_cost if current_output_cost is not None else "None"
                        new_val = change_info['changes']['output_cost']['new']

                        # Calculate percentage change
                        if current_output_cost is not None and current_output_cost != 0:
                            pct_change = ((new_val - current_output_cost) / current_output_cost) * 100
                            pct_str = f" ({pct_change:+.1f}%)"
                        else:
                            pct_str = ""

                        self.logger.info(f"  Output cost: {old_val} → {new_val}{pct_str}")

        if cost_changes:
            self.logger.info(f"Identified {len(cost_changes)} models with cost updates")
        else:
            self.logger.info("No cost updates needed - all costs are current")

        return config, cost_changes

    def preview_changes(self, invalid_models: List[Tuple[int, str, str]], provider_name: str) -> None:
        """Preview what changes would be made in dry-run mode."""
        provider = self.provider_manager.get_provider(provider_name)

        if not invalid_models:
            self.logger.info(f"[DRY-RUN] No invalid {provider.name} models found. No changes needed.")
            return

        self.logger.info(f"[DRY-RUN] Would remove the following invalid {provider.name} models:")
        for _, model_id, model_name in invalid_models:
            self.logger.info(f"  - Model: {model_id} (model_name: {model_name})")

        self.logger.info(f"[DRY-RUN] {len(invalid_models)} model entries would be removed from {self.config_path}")
        self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")

    def remove_invalid_entries(self, config: Dict[str, Any],
                             invalid_models: List[Tuple[int, str, str]]) -> Dict[str, Any]:
        """Remove invalid model entries from the configuration."""
        if not invalid_models:
            self.logger.info("No invalid models to remove")
            return config

        # Sort indices in descending order to avoid index shifting issues
        invalid_indices = sorted([index for index, _, _ in invalid_models], reverse=True)

        model_list = config['model_list']
        removed_count = 0

        for index in invalid_indices:
            if 0 <= index < len(model_list):
                removed_model = model_list.pop(index)
                removed_count += 1

                # Log the removal
                model_name = removed_model.get('model_name', 'unnamed')
                model_id = removed_model.get('litellm_params', {}).get('model', 'unknown')
                self.logger.info(f"Removed model entry: {model_id} (name: {model_name})")
            else:
                self.logger.warning(f"Invalid index {index} for model removal")

        self.logger.info(f"Successfully removed {removed_count} invalid model entries")
        return config

    def sort_model_list(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Sort the model list by model_name alphabetically, then by model under litellm_params."""
        if 'model_list' not in config or not config['model_list']:
            self.logger.info("No model list found or model list is empty")
            return config, False

        model_list = config['model_list']

        # Create tuples for comparison (original order)
        original_order = []
        for model in model_list:
            model_name = model.get('model_name', 'unnamed')
            litellm_model = model.get('litellm_params', {}).get('model', '')
            original_order.append((model_name, litellm_model))

        # Sort by model_name first, then by litellm_params.model
        sorted_model_list = sorted(
            model_list,
            key=lambda x: (
                x.get('model_name', 'unnamed').lower(),
                x.get('litellm_params', {}).get('model', '').lower()
            )
        )

        # Create tuples for comparison (sorted order)
        sorted_order = []
        for model in sorted_model_list:
            model_name = model.get('model_name', 'unnamed')
            litellm_model = model.get('litellm_params', {}).get('model', '')
            sorted_order.append((model_name, litellm_model))

        # Check if sorting actually changed the order
        was_sorted = original_order != sorted_order

        if was_sorted:
            config['model_list'] = sorted_model_list
            self.logger.info(f"Sorted {len(model_list)} models by model_name, then by litellm_params.model")
            self.logger.debug(f"Original order: {original_order[:5]}{'...' if len(original_order) > 5 else ''}")
            self.logger.debug(f"Sorted order: {sorted_order[:5]}{'...' if len(sorted_order) > 5 else ''}")
        else:
            self.logger.info("Model list is already sorted by model_name and litellm_params.model")

        return config, was_sorted

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the updated configuration back to the file."""
        if self.dry_run:
            self.logger.info("DRY RUN: Would save configuration to file")
            return

        try:
            self.logger.info(f"Saving configuration to {self.config_path}")

            # Create backup
            backup_path = self.config_path.with_suffix('.yaml.backup')
            if self.config_path.exists():
                self.config_path.rename(backup_path)
                self.logger.info(f"Created backup at {backup_path}")

            # Save new configuration
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)

            self.logger.info("Configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            # Restore backup if it exists
            if backup_path.exists():
                backup_path.rename(self.config_path)
                self.logger.info("Restored configuration from backup")
            raise

    def add_model_to_config(self, config: Dict[str, Any], provider_name: str,
                           model_ids: List[str], api_models: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        """Add one or more models to the configuration for a specific provider."""
        strategy = self.provider_manager.get_strategy(provider_name)
        provider = self.provider_manager.get_provider(provider_name)

        added_models = []
        failed_models = []

        # Get existing models once to check for duplicates
        existing_models = self.extract_provider_models(config, provider_name)
        existing_model_ids = []

        for _, full_model_id, _ in existing_models:
            model_entry = {'litellm_params': {'model': full_model_id}}
            model_id = strategy.extract_model_id(model_entry)
            if model_id:
                existing_model_ids.append(model_id)

        existing_names = [name for _, _, name in existing_models]

        # Process each model ID
        for model_id in model_ids:
            self.logger.info(f"Processing model: {model_id}")

            # Find the model in API data
            if model_id not in api_models:
                self.logger.error(f"Model '{model_id}' not found in {provider.name} API")
                failed_models.append(model_id)
                continue

            # Check if model already exists in config
            if model_id in existing_model_ids:
                self.logger.warning(f"Model '{model_id}' already exists in configuration")
                failed_models.append(model_id)
                continue

            # Get API model info and create model entry
            api_model_info = api_models[model_id]
            model_entry = strategy.create_model_entry(model_id, api_model_info)

            # Check for name conflicts and make unique if needed
            model_name = model_entry['model_name']
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1

            model_entry['model_name'] = model_name

            # Add to config
            if not self.dry_run:
                config['model_list'].append(model_entry)

            full_model_id = model_entry['litellm_params']['model']
            added_models.append(full_model_id)
            existing_model_ids.append(model_id)  # Update list to avoid duplicates in this batch
            existing_names.append(model_name)    # Update list to avoid name conflicts in this batch

            self.logger.info(f"Added model '{full_model_id}' with name '{model_name}'")

            # Log costs
            input_cost = model_entry['litellm_params'].get('input_cost_per_token')
            output_cost = model_entry['litellm_params'].get('output_cost_per_token')
            if input_cost is not None:
                self.logger.info(f"  Input cost: {input_cost}")
            if output_cost is not None:
                self.logger.info(f"  Output cost: {output_cost}")

        # Report summary
        if failed_models:
            self.logger.warning(f"Failed to add {len(failed_models)} model(s): {', '.join(failed_models)}")
        if added_models:
            self.logger.info(f"Successfully processed {len(added_models)} model(s) for addition")

        return config, added_models

    def cleanup_provider(self, provider_name: str, add_models: Optional[List[str]] = None) -> Tuple[int, int, List[str]]:
        """Clean up models for a specific provider."""
        try:
            # Load configuration
            config = self.load_config()

            # Handle add-models functionality
            if add_models:
                # Fetch available models with pricing from API
                available_models = self.fetch_available_models(provider_name)

                added_models = []
                if self.dry_run:
                    # Preview mode for adding models
                    existing_models = self.extract_provider_models(config, provider_name)
                    strategy = self.provider_manager.get_strategy(provider_name)

                    for model_id in add_models:
                        if model_id not in available_models:
                            self.logger.error(f"[DRY-RUN] Model '{model_id}' not found in {provider_name} API")
                            continue

                        # Check for duplicates
                        model_exists = False
                        for _, full_model_id, _ in existing_models:
                            model_entry = {'litellm_params': {'model': full_model_id}}
                            existing_id = strategy.extract_model_id(model_entry)
                            if existing_id == model_id:
                                model_exists = True
                                break

                        if model_exists:
                            self.logger.warning(f"[DRY-RUN] Model '{model_id}' already exists in configuration")
                        else:
                            api_model_info = available_models[model_id]
                            model_entry = strategy.create_model_entry(model_id, api_model_info)
                            self.logger.info(f"[DRY-RUN] Would add model '{model_entry['litellm_params']['model']}' with name '{model_entry['model_name']}'")
                            
                            # Show cost information in dry run
                            litellm_params = model_entry.get('litellm_params', {})
                            input_cost = litellm_params.get('input_cost_per_token')
                            output_cost = litellm_params.get('output_cost_per_token')
                            if input_cost is not None and output_cost is not None:
                                self.logger.info(f"[DRY-RUN]   Input cost: {input_cost}, Output cost: {output_cost}")
                            else:
                                self.logger.warning(f"[DRY-RUN]   Cost information missing: input={input_cost}, output={output_cost}")
                            
                            added_models.append(model_id)

                    if added_models:
                        self.logger.info(f"[DRY-RUN] Would add {len(added_models)} model(s): {', '.join(added_models)}")
                    else:
                        self.logger.info("[DRY-RUN] No valid models to add.")
                    return 0, 0, [f"Previewed {len(added_models)} models for addition"]
                else:
                    # Actually add models
                    config, added_models = self.add_model_to_config(config, provider_name, add_models, available_models)

                    if added_models:
                        # Sort the model list after adding new models
                        config, was_sorted = self.sort_model_list(config)
                        self.save_config(config)
                        self.logger.info(f"✅ Successfully added {len(added_models)} model(s): {', '.join(added_models)}")
                        if was_sorted:
                            self.logger.info("✅ Model list sorted by model_name")
                    else:
                        self.logger.warning("⚠️ No models were added - all models either already exist or were not found in API")

                return 0, 0, [f"Added {len(added_models)} model(s)"]

            # Sort the model list first
            config, was_sorted = self.sort_model_list(config)

            # Fetch available models with pricing from API
            available_models = self.fetch_available_models(provider_name)

            # Extract provider models for cleanup/validation
            provider_models = self.extract_provider_models(config, provider_name)

            if not provider_models:
                if self.dry_run:
                    self.logger.info(f"[DRY-RUN] No {provider_name} models found to process")
                else:
                    if was_sorted:
                        self.save_config(config)
                    self.logger.info(f"No {provider_name} models found to process")
                return 0, 0, []

            # Validate models
            invalid_models = self.validate_models(provider_models, available_models, provider_name)

            # Validate and update costs
            config, cost_changes = self.validate_and_update_costs(config, provider_models, available_models, provider_name)

            if self.dry_run:
                # Preview mode - show what would be changed
                self.preview_changes(invalid_models, provider_name)
            else:
                # Actually apply changes
                changes_made = was_sorted

                # Remove invalid entries
                if invalid_models:
                    config = self.remove_invalid_entries(config, invalid_models)
                    changes_made = True

                # Save config if any changes were made (sorting, invalid models, or cost updates)
                if was_sorted or invalid_models or cost_changes:
                    self.save_config(config)
                    changes_made = True

                if not changes_made:
                    self.logger.info(f"No changes needed for {provider_name} models")

            # Generate summary
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

    def cleanup_all_providers(self, add_models: Optional[Dict[str, List[str]]] = None) -> Dict[str, Tuple[int, int, List[str]]]:
        """Clean up models for all configured providers."""
        results = {}

        for provider_name in self.provider_names:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing provider: {provider_name}")
            self.logger.info(f"{'='*60}")

            provider_add_models = add_models.get(provider_name) if add_models else None
            try:
                models_removed, models_updated, changes_made = self.cleanup_provider(provider_name, provider_add_models)
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
1. Sorts the model list alphabetically by model_name, then by litellm_params.model
2. Validates models against the current API and removes invalid entries (except special models)
3. Updates model costs (input_cost_per_token/output_cost_per_token) when they differ from API pricing
4. Adds one or more models to the configuration

Supported providers: openrouter, requesty, novita, nano_gpt, all

API Key Requirements:
  - Requesty: Set REQUESTY_API_KEY environment variable
  - Nano-GPT: Set NANOGPT_API_KEY environment variable
  - OpenRouter: No API key required for model listing

Examples:
  %(prog)s --provider openrouter                           # Process OpenRouter models
  %(prog)s --provider all                                   # Process all providers
  %(prog)s --provider requesty --add-model "model1 model2"  # Add models to Requesty
  %(prog)s --provider openrouter --config my-config.yaml   # Use custom config
  %(prog)s --provider novita --dry-run --verbose           # Preview changes
        """
    )

    parser.add_argument(
        '--provider', '-p',
        required=True,
        help='Provider to process (openrouter, requesty, novita, nano_gpt, or "all")'
    )

    parser.add_argument(
        '--config', '-c',
        default="config.yaml",
        help='Path to LiteLLM configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Preview all changes without modifying the configuration file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )

    parser.add_argument(
        '--add-model', '-a',
        help='Add model(s) to configuration. Format: "model1 model2" or provider:model1,provider:model2'
    )

    args = parser.parse_args()

    try:
        load_dotenv(override=True)  # Load environment variables from .env file if present

        # Initialize provider manager to get available providers
        temp_manager = ProviderManager()
        available_providers = temp_manager.list_providers()

        # Determine which providers to process
        if args.provider.lower() == 'all':
            provider_names = available_providers
        else:
            provider_names = [args.provider.lower()]

        # Validate provider names
        for provider_name in provider_names:
            if provider_name not in available_providers:
                print(f"Error: Unknown provider '{provider_name}'. Available providers: {', '.join(available_providers)}", file=sys.stderr)
                return 1

        # Parse add-model argument
        add_models = None
        if args.add_model:
            add_models = {}

            # Parse provider|model format or default to first provider
            model_specs = args.add_model.split()
            for model_spec in model_specs:
                if '|' in model_spec:
                    # provider|model format
                    provider, model_id = model_spec.split('|', 1)
                    if provider not in provider_names:
                        print(f"Warning: Provider '{provider}' not in selected providers, skipping model '{model_id}'", file=sys.stderr)
                        continue
                else:
                    # Default to first provider if only one provider specified
                    if len(provider_names) == 1:
                        provider = provider_names[0]
                        model_id = model_spec
                    else:
                        print(f"Error: Must specify provider for model '{model_spec}' when using multiple providers. Use format provider|model", file=sys.stderr)
                        return 1

                if provider not in add_models:
                    add_models[provider] = []
                add_models[provider].append(model_id)

        # Create and run the cleaner
        cleaner = UnifiedModelCleaner(
            config_path=args.config,
            provider_names=provider_names,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        if len(provider_names) == 1:
            # Single provider mode
            provider_add_models = add_models.get(provider_names[0]) if add_models else None
            models_removed, models_updated, changes_made = cleaner.cleanup_provider(provider_names[0], provider_add_models)

            print(f"\nSummary for {provider_names[0]}:")
            print(f"Models removed: {models_removed}")
            print(f"Models updated: {models_updated}")
            print(f"Total changes: {len(changes_made)}")

            if changes_made:
                print(f"\nChanges made:")
                for change in changes_made:
                    print(f"  - {change}")
        else:
            # Multiple providers mode
            results = cleaner.cleanup_all_providers(add_models)

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
            print(f"Total changes across all providers: {total_changes}")

        if args.dry_run:
            print(f"\nDRY RUN: No actual changes were made")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())