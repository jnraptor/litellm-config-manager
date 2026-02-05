#!/usr/bin/env python3
"""
Base classes and utilities for LiteLLM model cleanup scripts.

This module provides common functionality shared across all provider-specific
cleanup scripts, including:
- Logging setup
- YAML config loading/saving
- Cost comparison utilities
- Model list sorting
- Preview and report generation
- Common CLI argument handling

Usage:
    from cleanup_base import BaseModelCleaner, setup_common_args

Author: LiteLLM Config Management
"""

import argparse
import logging
import sys
import time
import yaml
import requests
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path


__all__ = [
    'DEFAULT_CONFIG_FILE',
    'setup_logging',
    'costs_are_equal',
    'adjust_cost_for_free_model',
    'BaseModelCleaner',
    'setup_common_args',
    'validate_model_name_arg',
    'fetch_models_from_api',
    'APIClient',
]

DEFAULT_CONFIG_FILE = "config.yaml"


def setup_logging(verbose: bool = False, name: str = __name__) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


def costs_are_equal(cost1: float, cost2: float, rel_tol: float = 1e-9) -> bool:
    """
    Compare two cost values using relative tolerance for scientific notation.
    
    Properly handles very small cost values in scientific notation
    (e.g., 6.0e-07 vs 4e-07).
    
    Args:
        cost1: First cost value
        cost2: Second cost value
        rel_tol: Relative tolerance (default: 1e-9)
        
    Returns:
        True if costs are equal within tolerance, False otherwise
    """
    if cost1 == cost2:
        return True
    
    if cost1 == 0.0 or cost2 == 0.0:
        return cost1 == cost2
    
    abs_diff = abs(cost1 - cost2)
    max_abs = max(abs(cost1), abs(cost2))
    return abs_diff <= rel_tol * max_abs


def adjust_cost_for_free_model(cost: Optional[float], free_cost: float = 1e-09) -> Optional[float]:
    """
    Adjust cost for free models (0.0 -> 1e-09 for LiteLLM compatibility).
    
    Args:
        cost: The cost value from API
        free_cost: The nominal cost to use for free models
        
    Returns:
        Adjusted cost value
    """
    if cost is None:
        return None
    return free_cost if cost == 0.0 else cost


class APIClient:
    """
    HTTP client for API requests with retry logic and caching.
    
    Provides exponential backoff retry for transient failures and
    optional response caching for repeated requests.
    """
    
    def __init__(self, timeout: int = 30, max_retries: int = 3, 
                 base_delay: float = 1.0, use_cache: bool = True):
        """
        Initialize the API client.
        
        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            use_cache: Whether to cache responses
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.use_cache = use_cache
        self.session = requests.Session()
        self._cache: Dict[str, Any] = {}
    
    def _get_cache_key(self, url: str, headers: Optional[Dict[str, str]]) -> str:
        """Generate a cache key from URL and headers."""
        header_str = str(sorted(headers.items())) if headers else ""
        return f"{url}|{header_str}"
    
    def fetch(self, url: str, headers: Optional[Dict[str, str]] = None,
              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
        """
        Fetch JSON data from URL with retry logic.
        
        Args:
            url: The URL to fetch
            headers: Optional request headers
            logger: Optional logger for debug output
            
        Returns:
            Parsed JSON response
            
        Raises:
            requests.RequestException: If all retries fail
        """
        cache_key = self._get_cache_key(url, headers)
        
        if self.use_cache and cache_key in self._cache:
            if logger:
                logger.debug(f"Using cached response for {url}")
            return self._cache[cache_key]
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                
                if self.use_cache:
                    self._cache[cache_key] = data
                
                return data
                
            except requests.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    if logger:
                        logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                        logger.warning(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        
        raise last_exception
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()


# Global API client instance for module-level functions
_api_client = APIClient()


class BaseModelCleaner(ABC):
    """
    Abstract base class for model cleanup operations.
    
    Provides common functionality for loading/saving configs, sorting models,
    validating costs, and generating reports.
    """
    
    # Override these in subclasses
    PROVIDER_NAME: str = "base"
    API_URL: str = ""
    MODEL_PREFIX: str = ""
    SPECIAL_MODELS: Set[str] = set()
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = setup_logging(verbose, f"{self.__class__.__name__}")
    
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
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the updated configuration back to the file."""
        if self.dry_run:
            self.logger.info("DRY RUN: Would save configuration to file")
            return
        
        try:
            self.logger.info(f"Saving updated configuration to {self.config_path}")
            
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False,
                         allow_unicode=True, width=1000)
            
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise

    def sort_model_list(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Sort the model list by model_name alphabetically, then by litellm_params.model.
        
        Args:
            config: The configuration dictionary
            
        Returns:
            Tuple of (updated_config, was_sorted)
        """
        if 'model_list' not in config or not config['model_list']:
            self.logger.info("No model list found or model list is empty")
            return config, False
        
        model_list = config['model_list']
        
        original_order = [
            (m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', ''))
            for m in model_list
        ]
        
        sorted_model_list = sorted(
            model_list,
            key=lambda x: (
                x.get('model_name', 'unnamed').lower(),
                x.get('litellm_params', {}).get('model', '').lower()
            )
        )
        
        sorted_order = [
            (m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', ''))
            for m in sorted_model_list
        ]
        
        was_sorted = original_order != sorted_order
        
        if was_sorted:
            config['model_list'] = sorted_model_list
            self.logger.info(f"Sorted {len(model_list)} models by model_name, then by litellm_params.model")
        else:
            self.logger.info("Model list is already sorted by model_name and litellm_params.model")
        
        return config, was_sorted
    
    def remove_invalid_entries(self, config: Dict[str, Any],
                              invalid_models: List[Tuple[int, str, str]]) -> Dict[str, Any]:
        """
        Remove invalid model entries from the configuration.
        
        Args:
            config: The configuration dictionary
            invalid_models: List of (index, model_id, model_name) to remove
            
        Returns:
            Updated configuration dictionary
        """
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
            else:
                self.logger.warning(f"Invalid index {index} for model removal")
        
        self.logger.info(f"Successfully removed {removed_count} invalid model entries")
        return config
    
    def preview_changes(self, invalid_models: List[Tuple[int, str, str]]) -> None:
        """Preview what changes would be made in dry-run mode."""
        if not invalid_models:
            self.logger.info(f"[DRY-RUN] No invalid {self.PROVIDER_NAME} models found. No changes needed.")
            return
        
        self.logger.info(f"[DRY-RUN] Would remove the following invalid {self.PROVIDER_NAME} models:")
        for _, model_id, model_name in invalid_models:
            self.logger.info(f"  - Model: {model_id} (model_name: {model_name})")
        
        self.logger.info(f"[DRY-RUN] {len(invalid_models)} model entries would be removed from {self.config_path}")
        self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
    
    def preview_cost_changes(self, cost_changes: List[Dict[str, Any]]) -> None:
        """Preview cost changes in dry-run mode."""
        if not cost_changes:
            self.logger.info("[DRY-RUN] No cost updates needed.")
            return
        
        self.logger.info(f"[DRY-RUN] Would update costs for {len(cost_changes)} models:")
        
        for change in cost_changes:
            model_id = change['model_id']
            model_name = change['model_name']
            changes = change['changes']
            
            self.logger.info(f"  - Model: {model_id} (name: {model_name})")
            
            for cost_type in ['input_cost', 'output_cost']:
                if cost_type in changes:
                    old_val = changes[cost_type]['old']
                    new_val = changes[cost_type]['new']
                    old_str = str(old_val) if old_val is not None else "None"
                    
                    pct_str = ""
                    if old_val is not None and old_val != 0:
                        pct_change = ((new_val - old_val) / old_val) * 100
                        pct_str = f" ({pct_change:+.1f}%)"
                    
                    cost_label = "Input cost" if cost_type == 'input_cost' else "Output cost"
                    self.logger.info(f"    {cost_label}: {old_str} → {new_val}{pct_str}")

    def preview_sort_changes(self, config: Dict[str, Any]) -> None:
        """Preview what the sorting would change in dry-run mode."""
        if 'model_list' not in config or not config['model_list']:
            self.logger.info("[DRY-RUN] No model list found or model list is empty")
            return
        
        model_list = config['model_list']
        
        original_order = [
            (m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', ''))
            for m in model_list
        ]
        
        sorted_model_list = sorted(
            model_list,
            key=lambda x: (
                x.get('model_name', 'unnamed').lower(),
                x.get('litellm_params', {}).get('model', '').lower()
            )
        )
        
        sorted_order = [
            (m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', ''))
            for m in sorted_model_list
        ]
        
        would_sort = original_order != sorted_order
        
        if would_sort:
            self.logger.info(f"[DRY-RUN] Would sort {len(model_list)} models by model_name, then by litellm_params.model")
            self.logger.info("[DRY-RUN] Current order (first 10):")
            for i, (name, model) in enumerate(original_order[:10]):
                self.logger.info(f"[DRY-RUN]   {i+1:2d}. {name} ({model})")
            if len(original_order) > 10:
                self.logger.info(f"[DRY-RUN]   ... and {len(original_order) - 10} more")
            
            self.logger.info("[DRY-RUN] Would become (first 10):")
            for i, (name, model) in enumerate(sorted_order[:10]):
                self.logger.info(f"[DRY-RUN]   {i+1:2d}. {name} ({model})")
            if len(sorted_order) > 10:
                self.logger.info(f"[DRY-RUN]   ... and {len(sorted_order) - 10} more")
        else:
            self.logger.info("[DRY-RUN] Model list is already sorted - no changes needed")
    
    def generate_report(self, invalid_models: List[Tuple[int, str, str]],
                       cost_changes: Optional[List[Dict[str, Any]]] = None,
                       was_sorted: bool = False) -> None:
        """Generate a summary report of the cleanup operation."""
        if cost_changes is None:
            cost_changes = []
        
        if invalid_models:
            if self.dry_run:
                self.logger.info(f"📋 [DRY-RUN] Summary: {len(invalid_models)} invalid models identified")
            else:
                self.logger.info(f"✅ Model cleanup: {len(invalid_models)} invalid models removed")
            
            for _, model_id, model_name in invalid_models:
                status = "[WOULD REMOVE]" if self.dry_run else "[REMOVED]"
                self.logger.info(f"  {status} {model_id} (name: {model_name})")
        
        if cost_changes:
            if self.dry_run:
                self.logger.info(f"💰 [DRY-RUN] Cost updates: {len(cost_changes)} models would have cost changes")
            else:
                self.logger.info(f"✅ Cost updates: {len(cost_changes)} models had cost changes applied")
        
        if was_sorted:
            if self.dry_run:
                self.logger.info("📝 [DRY-RUN] Model list would be sorted by model_name")
            else:
                self.logger.info("✅ Model list sorted by model_name")
        
        if not invalid_models and not cost_changes and not was_sorted:
            self.logger.info(f"✅ All {self.PROVIDER_NAME} models are valid with current costs and list is already sorted")
        elif self.dry_run:
            total_changes = len(invalid_models) + len(cost_changes) + (1 if was_sorted else 0)
            self.logger.info(f"📋 [DRY-RUN] Total changes identified: {total_changes}")
            self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
        else:
            total_changes = len(invalid_models) + len(cost_changes) + (1 if was_sorted else 0)
            self.logger.info(f"✅ Cleanup completed: {total_changes} total changes applied")
    
    @abstractmethod
    def extract_provider_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """
        Extract models for this provider from the configuration.
        
        Returns:
            List of tuples: (index, model_id, model_name)
        """
        pass
    
    @abstractmethod
    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch available models with pricing from provider API.
        
        Returns:
            Dict mapping model_id to model info including costs
        """
        pass
    
    @abstractmethod
    def get_api_model_id(self, model_id: str) -> str:
        """
        Extract the API model ID from the config model ID.
        
        Args:
            model_id: The model ID from config (e.g., "openrouter/model-name")
            
        Returns:
            The API model ID (e.g., "model-name")
        """
        pass

    def validate_models(self, config_models: List[Tuple[int, str, str]],
                       api_models: Dict[str, Dict[str, Any]]) -> List[Tuple[int, str, str]]:
        """
        Compare config models with API models and identify invalid ones.
        
        Args:
            config_models: List of (index, model_id, model_name) from config
            api_models: Dict of model data from API with pricing info
            
        Returns:
            List of invalid models: (index, model_id, model_name)
        """
        api_models_set = set(api_models.keys())
        invalid_models = []
        
        for index, model_id, model_name in config_models:
            api_model_id = self.get_api_model_id(model_id)
            
            if api_model_id in self.SPECIAL_MODELS:
                self.logger.debug(f"Special model found (will not be removed): {model_id} -> {api_model_id}")
                continue
            
            if api_model_id not in api_models_set:
                invalid_models.append((index, model_id, model_name))
                self.logger.debug(f"Invalid model found: {model_id} -> {api_model_id}")
            else:
                self.logger.debug(f"Valid model: {model_id} -> {api_model_id}")
        
        self.logger.info(f"Identified {len(invalid_models)} invalid {self.PROVIDER_NAME} models")
        return invalid_models
    
    def validate_and_update_costs(self, config: Dict[str, Any],
                                 config_models: List[Tuple[int, str, str]],
                                 api_models: Dict[str, Dict[str, Any]],
                                 provider_order: int = 2) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Validate and update model costs based on API pricing.
        
        Args:
            config: The configuration dictionary
            config_models: List of (index, model_id, model_name) from config
            api_models: Dict of model data from API with pricing info
            provider_order: The order value to set for models from this provider
            
        Returns:
            Tuple of (updated_config, list_of_cost_changes)
        """
        cost_changes = []
        model_list = config['model_list']
        
        for index, model_id, model_name in config_models:
            api_model_id = self.get_api_model_id(model_id)
            
            if api_model_id not in api_models:
                continue
            
            api_model_info = api_models[api_model_id]
            api_input_cost = api_model_info.get('input_cost')
            api_output_cost = api_model_info.get('output_cost')
            
            if 0 <= index < len(model_list):
                model_entry = model_list[index]
                litellm_params = model_entry.get('litellm_params', {})
                current_input_cost = litellm_params.get('input_cost_per_token')
                current_output_cost = litellm_params.get('output_cost_per_token')
                
                input_changed = False
                output_changed = False
                change_info = {
                    'index': index,
                    'model_id': model_id,
                    'model_name': model_name,
                    'changes': {}
                }
                
                # Compare input costs
                if api_input_cost is not None:
                    adjusted_input_cost = adjust_cost_for_free_model(api_input_cost)
                    
                    if current_input_cost is not None:
                        if not costs_are_equal(current_input_cost, adjusted_input_cost):
                            input_changed = True
                    else:
                        input_changed = True
                    
                    if input_changed:
                        change_info['changes']['input_cost'] = {
                            'old': current_input_cost,
                            'new': adjusted_input_cost
                        }
                        litellm_params['input_cost_per_token'] = adjusted_input_cost
                        self.logger.debug(f"Input cost change for {model_id}: {current_input_cost} → {adjusted_input_cost}")
                
                # Compare output costs
                if api_output_cost is not None:
                    adjusted_output_cost = adjust_cost_for_free_model(api_output_cost)
                    
                    if current_output_cost is not None:
                        if not costs_are_equal(current_output_cost, adjusted_output_cost):
                            output_changed = True
                    else:
                        output_changed = True
                    
                    if output_changed:
                        change_info['changes']['output_cost'] = {
                            'old': current_output_cost,
                            'new': adjusted_output_cost
                        }
                        litellm_params['output_cost_per_token'] = adjusted_output_cost
                        self.logger.debug(f"Output cost change for {model_id}: {current_output_cost} → {adjusted_output_cost}")
                
                # Update order value from provider configuration
                litellm_params['order'] = provider_order
                
                if input_changed or output_changed:
                    cost_changes.append(change_info)
                    self._log_cost_change(model_id, model_name, change_info, 
                                         current_input_cost, current_output_cost)
        
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

    def generate_model_name(self, model_id: str, prefix: str = "") -> str:
        """
        Generate an appropriate model_name from the model ID.
        
        Args:
            model_id: The model ID (e.g., "provider/model-name")
            prefix: Optional prefix to add (e.g., "or-" for OpenRouter)
            
        Returns:
            Generated model name
        """
        clean_id = model_id.replace('/', '-').replace(':', '-')
        model_name = f"{prefix}{clean_id}" if prefix else clean_id
        return model_name.lower()
    
    def find_model_in_api(self, model_id: str, 
                         api_models: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find a specific model in the API models data."""
        return api_models.get(model_id)
    
    def add_model_to_config(self, config: Dict[str, Any], model_ids: List[str],
                           api_models: Dict[str, Dict[str, Any]],
                           custom_model_name: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Add one or more models to the configuration.
        
        Args:
            config: The configuration dictionary
            model_ids: List of model IDs to add
            api_models: Dict of available models from API
            custom_model_name: Optional custom name (only for single model)
            
        Returns:
            Tuple of (updated_config, list_of_added_model_ids)
        """
        added_models = []
        failed_models = []
        
        existing_models = self.extract_provider_models(config)
        existing_model_ids = [self.get_api_model_id(mid) for _, mid, _ in existing_models]
        existing_names = [name for _, _, name in existing_models]
        
        for model_id in model_ids:
            self.logger.info(f"Processing model: {model_id}")
            
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                self.logger.error(f"Model '{model_id}' not found in {self.PROVIDER_NAME} API")
                failed_models.append(model_id)
                continue
            
            if model_id in existing_model_ids:
                self.logger.warning(f"Model '{model_id}' already exists in configuration")
                failed_models.append(model_id)
                continue
            
            # Generate model name
            if custom_model_name and len(model_ids) == 1:
                model_name = custom_model_name
            else:
                model_name = self.generate_model_name(model_id)
            
            # Handle name conflicts
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1
            
            # Create model entry (subclasses should override create_model_entry)
            model_entry = self.create_model_entry(model_id, model_info, model_name)
            
            config['model_list'].append(model_entry)
            added_models.append(model_id)
            existing_model_ids.append(model_id)
            existing_names.append(model_name)
            
            self.logger.info(f"Added model '{model_id}' with name '{model_name}'")
            
            input_cost = model_entry.get('litellm_params', {}).get('input_cost_per_token')
            output_cost = model_entry.get('litellm_params', {}).get('output_cost_per_token')
            if input_cost is not None:
                self.logger.info(f"  Input cost: {input_cost}")
            if output_cost is not None:
                self.logger.info(f"  Output cost: {output_cost}")
        
        if failed_models:
            self.logger.warning(f"Failed to add {len(failed_models)} model(s): {', '.join(failed_models)}")
        if added_models:
            self.logger.info(f"Successfully processed {len(added_models)} model(s) for addition")
        
        return config, added_models
    
    @abstractmethod
    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any],
                          model_name: str) -> Dict[str, Any]:
        """
        Create a new model entry for the config.
        
        Args:
            model_id: The API model ID
            api_model_info: Model info from API including costs
            model_name: The model name to use
            
        Returns:
            Dict representing the model entry for config
        """
        pass
    
    def preview_add_model(self, model_ids: List[str], api_models: Dict[str, Dict[str, Any]],
                         custom_model_name: Optional[str] = None) -> None:
        """Preview what would be added when adding models."""
        config = self.load_config()
        existing_models = self.extract_provider_models(config)
        existing_model_ids = [self.get_api_model_id(mid) for _, mid, _ in existing_models]
        existing_names = [name for _, _, name in existing_models]
        
        valid_models = []
        invalid_models = []
        duplicate_models = []
        
        for model_id in model_ids:
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                invalid_models.append(model_id)
                continue
            
            if model_id in existing_model_ids:
                duplicate_models.append(model_id)
                continue
            
            if custom_model_name and len(model_ids) == 1:
                model_name = custom_model_name
            else:
                model_name = self.generate_model_name(model_id)
            
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1
            
            model_preview = {
                'id': model_id,
                'name': model_name,
                'input_cost': adjust_cost_for_free_model(model_info.get('input_cost')),
                'output_cost': adjust_cost_for_free_model(model_info.get('output_cost'))
            }
            
            valid_models.append(model_preview)
            existing_names.append(model_name)
        
        if invalid_models:
            self.logger.error(f"[DRY-RUN] {len(invalid_models)} model(s) not found in {self.PROVIDER_NAME} API: {', '.join(invalid_models)}")
        
        if duplicate_models:
            self.logger.warning(f"[DRY-RUN] {len(duplicate_models)} model(s) already exist in configuration: {', '.join(duplicate_models)}")
        
        if valid_models:
            self.logger.info(f"[DRY-RUN] Would add {len(valid_models)} model(s):")
            for model in valid_models:
                self.logger.info(f"[DRY-RUN]   - Model '{model['id']}' with name '{model['name']}'")
                self.logger.info(f"[DRY-RUN]     Input cost: {model['input_cost']}")
                self.logger.info(f"[DRY-RUN]     Output cost: {model['output_cost']}")
        else:
            self.logger.info("[DRY-RUN] No valid models to add.")


def setup_common_args(parser: argparse.ArgumentParser, default_config: str = DEFAULT_CONFIG_FILE) -> None:
    """
    Add common CLI arguments to an argument parser.
    
    Args:
        parser: The argument parser to add arguments to
        default_config: Default config file path
    """
    parser.add_argument(
        '--config', '-c',
        default=default_config,
        help=f'Configuration file path (default: {default_config})'
    )
    
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Show what would be changed without making actual changes'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--add-model', '-a',
        nargs='+',
        help='Add new model(s) to configuration (space-separated list)'
    )
    
    parser.add_argument(
        '--model-name',
        help='Custom model name to use when adding a single model. Only valid when --add-model is used with exactly one model.'
    )


def validate_model_name_arg(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """
    Validate the --model-name argument usage.
    
    Args:
        args: Parsed arguments
        parser: The argument parser (for error reporting)
    """
    if args.model_name:
        if not args.add_model:
            parser.error("--model-name can only be used with --add-model")
        if len(args.add_model) > 1:
            parser.error("--model-name can only be used when adding a single model")


def fetch_models_from_api(api_url: str, logger: logging.Logger,
                         headers: Optional[Dict[str, str]] = None,
                         timeout: int = 30) -> Dict[str, Any]:
    """
    Fetch models from an API endpoint.
    
    Args:
        api_url: The API URL to fetch from
        logger: Logger instance
        headers: Optional request headers
        timeout: Request timeout in seconds
        
    Returns:
        The JSON response data
        
    Raises:
        requests.RequestException: If the request fails
        ValueError: If the response format is invalid
    """
    logger.info(f"Fetching models from API: {api_url}")
    
    # Use the global API client with retry logic
    client = APIClient(timeout=timeout, use_cache=True)
    data = client.fetch(api_url, headers=headers, logger=logger)
    
    if 'data' not in data:
        raise ValueError("Invalid API response format: missing 'data' field")
    
    return data
