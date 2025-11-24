#!/usr/bin/env python3
"""
Novita Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Novita models in a LiteLLM config.yaml file against
the current Novita API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing

The script fetches current pricing from https://api.novita.ai/v3/openai/models and automatically
updates any cost differences found in the configuration file.

Usage:
    python cleanup_novita_models.py [--config config.yaml] [--dry-run] [--verbose]

Author: Generated for LiteLLM Config Management
"""

import argparse
import logging
import sys
import yaml
import requests
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


# Constants
NOVITA_API_URL = "https://api.novita.ai/v3/openai/models"
DEFAULT_CONFIG_FILE = "config.yaml"


class NovitaModelCleaner:
    """Main class for cleaning up invalid Novita models from LiteLLM config."""
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = self._setup_logging()
        
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
    
    def extract_novita_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """
        Extract Novita models from the configuration.
        
        Returns:
            List of tuples: (index, model_id, model_name)
        """
        novita_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
                
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            model_name = model_entry.get('model_name', 'unnamed')
            
            # Check if this is a Novita model
            if model_id.startswith('novita/'):
                novita_models.append((index, model_id, model_name))
                self.logger.debug(f"Found Novita model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(novita_models)} Novita models in configuration")
        return novita_models
    
    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch the list of available models with pricing from Novita API."""
        try:
            self.logger.info("Fetching available models with pricing from Novita API...")
            
            response = requests.get(NOVITA_API_URL, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("Invalid API response format: missing 'data' field")
            
            # Extract model IDs and pricing from the API response
            available_models = {}
            for model in data['data']:
                if isinstance(model, dict) and 'id' in model:
                    model_id = model['id']
                    model_info = {
                        'id': model_id,
                        'input_cost': None,
                        'output_cost': None
                    }
                    
                    # Extract pricing information from Novita API format
                    # input_token_price_per_m of 600 means $0.06 per million input tokens
                    # which would be 6.0e-08 input cost per token (0.06 / 1,000,000)
                    input_price_per_m = model.get('input_token_price_per_m')
                    if input_price_per_m is not None:
                        try:
                            # Convert from price per million to price per token
                            # input_price_per_m of 600 = $0.06 per million tokens
                            # So: (600 / 10000) / 1,000,000 = 0.06 / 1,000,000 = 6e-08
                            model_info['input_cost'] = round(float(input_price_per_m) / 10000 / 1_000_000, 12)
                        except (ValueError, TypeError):
                            self.logger.debug(f"Invalid input price for {model_id}: {input_price_per_m}")
                    
                    output_price_per_m = model.get('output_token_price_per_m')
                    if output_price_per_m is not None:
                        try:
                            # Convert from price per million to price per token
                            # output_price_per_m of 900 = $0.09 per million tokens
                            # So: (900 / 10000) / 1,000,000 = 0.09 / 1,000,000 = 9e-08
                            model_info['output_cost'] = round(float(output_price_per_m) / 10000 / 1_000_000, 12)
                        except (ValueError, TypeError):
                            self.logger.debug(f"Invalid output price for {model_id}: {output_price_per_m}")
                    
                    available_models[model_id] = model_info
            
            self.logger.info(f"Fetched {len(available_models)} available models from Novita API")
            self.logger.debug(f"Sample models: {list(available_models.keys())[:5]}")
            
            return available_models
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching models from Novita API: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing API response: {e}")
            raise
    
    def _costs_are_equal(self, cost1: float, cost2: float, rel_tol: float = 1e-9) -> bool:
        """
        Compare two cost values using relative tolerance to handle scientific notation.
        
        Args:
            cost1: First cost value
            cost2: Second cost value
            rel_tol: Relative tolerance (default 1e-9 for very small numbers)
            
        Returns:
            True if costs are equal within tolerance, False otherwise
        """
        # Handle exact equality first
        if cost1 == cost2:
            return True
        
        # Handle zero cases
        if cost1 == 0.0 or cost2 == 0.0:
            return cost1 == cost2
        
        # Use relative tolerance comparison for non-zero values
        abs_diff = abs(cost1 - cost2)
        max_abs = max(abs(cost1), abs(cost2))
        
        return abs_diff <= rel_tol * max_abs

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
        # Convert API models to set for faster lookup
        api_models_set = set(api_models.keys())
        invalid_models = []
        
        for index, model_id, model_name in config_models:
            # Extract the model ID part after 'novita/'
            # Novita models in config are like: novita/deepseek/deepseek-r1-0528-qwen3-8b
            # API models are like: deepseek/deepseek-r1-0528-qwen3-8b
            if model_id.startswith('novita/'):
                api_model_id = model_id[len('novita/'):]  # Remove 'novita/' prefix
                
                if api_model_id not in api_models_set:
                    invalid_models.append((index, model_id, model_name))
                    self.logger.debug(f"Invalid model found: {model_id} -> {api_model_id}")
                else:
                    self.logger.debug(f"Valid model: {model_id} -> {api_model_id}")
        
        self.logger.info(f"Identified {len(invalid_models)} invalid Novita models")
        return invalid_models
    
    def validate_and_update_costs(self, config: Dict[str, Any],
                                 config_models: List[Tuple[int, str, str]],
                                 api_models: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Validate and update model costs based on API pricing.
        
        Args:
            config: The configuration dictionary
            config_models: List of (index, model_id, model_name) from config
            api_models: Dict of model data from API with pricing info
            
        Returns:
            Tuple of (updated_config, list_of_cost_changes)
        """
        cost_changes = []
        model_list = config['model_list']
        
        for index, model_id, model_name in config_models:
            # Extract the API model ID
            if model_id.startswith('novita/'):
                api_model_id = model_id[len('novita/'):]
                
                # Check if model exists in API data
                if api_model_id not in api_models:
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
                    
                    # Check for cost changes
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
                        # Handle free models: if API returns 0.0, use 1e-09 for LiteLLM compatibility
                        adjusted_input_cost = 1e-09 if api_input_cost == 0.0 else api_input_cost

                        # Compare costs using relative tolerance for scientific notation values
                        if current_input_cost is not None:
                            if not self._costs_are_equal(current_input_cost, adjusted_input_cost):
                                input_changed = True
                                change_info['changes']['input_cost'] = {
                                    'old': current_input_cost,
                                    'new': adjusted_input_cost
                                }
                                litellm_params['input_cost_per_token'] = adjusted_input_cost
                                self.logger.debug(f"Input cost change for {model_id}: {current_input_cost} → {adjusted_input_cost}")
                        else:
                            # Current cost is None, always update
                            input_changed = True
                            change_info['changes']['input_cost'] = {
                                'old': current_input_cost,
                                'new': adjusted_input_cost
                            }
                            litellm_params['input_cost_per_token'] = adjusted_input_cost
                            self.logger.debug(f"Input cost change for {model_id}: {current_input_cost} → {adjusted_input_cost}")

                    # Compare output costs
                    if api_output_cost is not None:
                        # Handle free models: if API returns 0.0, use 1e-09 for LiteLLM compatibility
                        adjusted_output_cost = 1e-09 if api_output_cost == 0.0 else api_output_cost

                        # Compare costs using relative tolerance for scientific notation values
                        if current_output_cost is not None:
                            if not self._costs_are_equal(current_output_cost, adjusted_output_cost):
                                output_changed = True
                                change_info['changes']['output_cost'] = {
                                    'old': current_output_cost,
                                    'new': adjusted_output_cost
                                }
                                litellm_params['output_cost_per_token'] = adjusted_output_cost
                                self.logger.debug(f"Output cost change for {model_id}: {current_output_cost} → {adjusted_output_cost}")
                        else:
                            # Current cost is None, always update
                            output_changed = True
                            change_info['changes']['output_cost'] = {
                                'old': current_output_cost,
                                'new': adjusted_output_cost
                            }
                            litellm_params['output_cost_per_token'] = adjusted_output_cost
                            self.logger.debug(f"Output cost change for {model_id}: {current_output_cost} → {adjusted_output_cost}")
                    
                    # Record changes if any occurred
                    if input_changed or output_changed:
                        cost_changes.append(change_info)
                        self.logger.info(f"Cost update for {model_id} (name: {model_name})")
                        
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
            
            if 'input_cost' in changes:
                old_val = changes['input_cost']['old']
                new_val = changes['input_cost']['new']
                old_str = str(old_val) if old_val is not None else "None"
                
                # Calculate percentage change
                if old_val is not None and old_val != 0:
                    pct_change = ((new_val - old_val) / old_val) * 100
                    pct_str = f" ({pct_change:+.1f}%)"
                else:
                    pct_str = ""
                
                self.logger.info(f"    Input cost: {old_str} → {new_val}{pct_str}")
            
            if 'output_cost' in changes:
                old_val = changes['output_cost']['old']
                new_val = changes['output_cost']['new']
                old_str = str(old_val) if old_val is not None else "None"
                
                # Calculate percentage change
                if old_val is not None and old_val != 0:
                    pct_change = ((new_val - old_val) / old_val) * 100
                    pct_str = f" ({pct_change:+.1f}%)"
                else:
                    pct_str = ""
                
                self.logger.info(f"    Output cost: {old_str} → {new_val}{pct_str}")
    
    def preview_changes(self, invalid_models: List[Tuple[int, str, str]]) -> None:
        """Preview what changes would be made in dry-run mode."""
        if not invalid_models:
            self.logger.info("[DRY-RUN] No invalid Novita models found. No changes needed.")
            return
        
        self.logger.info("[DRY-RUN] Would remove the following invalid Novita models:")
        for _, model_id, model_name in invalid_models:
            self.logger.info(f"  - Model: {model_id} (model_name: {model_name})")
        
        self.logger.info(f"[DRY-RUN] {len(invalid_models)} model entries would be removed from {self.config_path}")
        self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
    
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
        """
        Sort the model list by model_name alphabetically, then by model under litellm_params.
        
        Args:
            config: The configuration dictionary
            
        Returns:
            Tuple of (updated_config, was_sorted)
        """
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
    
    def preview_sort_changes(self, config: Dict[str, Any]) -> None:
        """Preview what the sorting would change in dry-run mode."""
        if 'model_list' not in config or not config['model_list']:
            self.logger.info("[DRY-RUN] No model list found or model list is empty")
            return
        
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
        
        # Check if sorting would change the order
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
            self.logger.info("[DRY-RUN] Model list is already sorted by model_name and litellm_params.model - no changes needed")

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the updated configuration back to the file."""
        try:
            self.logger.info(f"Saving updated configuration to {self.config_path}")
            
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False,
                         allow_unicode=True, width=1000)
            
            self.logger.info("Configuration saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def generate_report(self, invalid_models: List[Tuple[int, str, str]],
                       cost_changes: Optional[List[Dict[str, Any]]] = None,
                       was_sorted: bool = False) -> None:
        """Generate a summary report of the cleanup operation."""
        if cost_changes is None:
            cost_changes = []
        
        # Report on invalid models
        if invalid_models:
            if self.dry_run:
                self.logger.info(f"📋 [DRY-RUN] Summary: {len(invalid_models)} invalid models identified")
            else:
                self.logger.info(f"✅ Model cleanup: {len(invalid_models)} invalid models removed")
                
            # List the invalid models
            for _, model_id, model_name in invalid_models:
                status = "[WOULD REMOVE]" if self.dry_run else "[REMOVED]"
                self.logger.info(f"  {status} {model_id} (name: {model_name})")
        
        # Report on cost changes
        if cost_changes:
            if self.dry_run:
                self.logger.info(f"💰 [DRY-RUN] Cost updates: {len(cost_changes)} models would have cost changes")
            else:
                self.logger.info(f"✅ Cost updates: {len(cost_changes)} models had cost changes applied")
        
        # Report on sorting
        if was_sorted:
            if self.dry_run:
                self.logger.info("📝 [DRY-RUN] Model list would be sorted by model_name")
            else:
                self.logger.info("✅ Model list sorted by model_name")
        
        # Overall summary
        if not invalid_models and not cost_changes and not was_sorted:
            self.logger.info("✅ All Novita models are valid with current costs and list is already sorted")
        elif self.dry_run:
            total_changes = len(invalid_models) + len(cost_changes) + (1 if was_sorted else 0)
            self.logger.info(f"📋 [DRY-RUN] Total changes identified: {total_changes}")
            self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
        else:
            total_changes = len(invalid_models) + len(cost_changes) + (1 if was_sorted else 0)
            self.logger.info(f"✅ Cleanup completed: {total_changes} total changes applied")
    
    def generate_model_name(self, model_id: str) -> str:
        """
        Generate an appropriate model_name from the Novita model ID.
        
        Args:
            model_id: The Novita model ID (e.g., "deepseek/deepseek-r1-0528-qwen3-8b")
            
        Returns:
            Generated model name (e.g., "nv-deepseek-r1-0528-qwen3-8b")
        """
        # Remove common prefixes and clean up the name
        clean_id = model_id.replace('/', '-').replace(':', '-')
        
        # Add 'nv-' prefix for Novita models
        model_name = f"nv-{clean_id}"
        
        # Handle some common cases to make names more readable
        model_name = model_name.replace('deepseek-', '').replace('meta-llama-', '').replace('google-', '')
        model_name = model_name.replace('mistralai-', 'mistral-').replace('qwen-', '')
        
        self.logger.debug(f"Generated model name '{model_name}' from ID '{model_id}'")
        return model_name
    
    def find_model_in_api(self, model_id: str, api_models: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find a specific model in the API models data.
        
        Args:
            model_id: The model ID to search for
            api_models: Dict of model data from API
            
        Returns:
            Model info dict if found, None otherwise
        """
        if model_id in api_models:
            return api_models[model_id]
        
        return None
    
    def add_model_to_config(self, config: Dict[str, Any], model_ids: List[str], 
                           api_models: Dict[str, Dict[str, Any]], custom_model_name: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Add one or more Novita models to the configuration.
        
        Args:
            config: The configuration dictionary
            model_ids: List of Novita model IDs to add
            api_models: Dict of model data from API
            
        Returns:
            Tuple of (updated_config, list_of_added_models)
        """
        added_models = []
        failed_models = []
        
        # Get existing models once to check for duplicates
        existing_models = self.extract_novita_models(config)
        existing_model_ids = [mid.replace('novita/', '') for _, mid, _ in existing_models]
        existing_names = [name for _, _, name in existing_models]
        
        # Process each model ID
        for model_id in model_ids:
            self.logger.info(f"Processing model: {model_id}")
            
            # Find the model in API data
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                self.logger.error(f"Model '{model_id}' not found in Novita API")
                failed_models.append(model_id)
                continue
            
            # Check if model already exists in config
            if model_id in existing_model_ids:
                self.logger.warning(f"Model '{model_id}' already exists in configuration")
                failed_models.append(model_id)
                continue
            
            # Generate model name
            if custom_model_name and len(model_ids) == 1:
                model_name = custom_model_name
            else:
                model_name = self.generate_model_name(model_id)
            
            # Check for name conflicts and make unique if needed
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1
            
            # Create model entry
            input_cost = model_info.get('input_cost')
            output_cost = model_info.get('output_cost')
            
            # Handle free models: if API returns 0.0, use 1e-09 for LiteLLM compatibility
            if input_cost is not None:
                input_cost = 1e-09 if input_cost == 0.0 else input_cost
            if output_cost is not None:
                output_cost = 1e-09 if output_cost == 0.0 else output_cost
            
            model_entry = {
                'litellm_params': {
                    'model': f'novita/{model_id}'
                },
                'model_name': model_name
            }
            
            # Add costs if available
            if input_cost is not None:
                model_entry['litellm_params']['input_cost_per_token'] = input_cost
            if output_cost is not None:
                model_entry['litellm_params']['output_cost_per_token'] = output_cost
            
            # Add to config
            config['model_list'].append(model_entry)
            added_models.append(model_id)
            existing_model_ids.append(model_id)  # Update list to avoid duplicates in this batch
            existing_names.append(model_name)    # Update list to avoid name conflicts in this batch
            
            self.logger.info(f"Added model '{model_id}' with name '{model_name}'")
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
    
    def preview_add_model(self, model_ids: List[str], api_models: Dict[str, Dict[str, Any]], custom_model_name: Optional[str] = None) -> None:
        """Preview what would be added when adding one or more models."""
        # Get existing models once to check for duplicates
        config = self.load_config()
        existing_models = self.extract_novita_models(config)
        existing_model_ids = [mid.replace('novita/', '') for _, mid, _ in existing_models]
        existing_names = [name for _, _, name in existing_models]
        
        valid_models = []
        invalid_models = []
        duplicate_models = []
        
        # Process each model ID for preview
        for model_id in model_ids:
            # Check if model exists in API
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                invalid_models.append(model_id)
                continue
            
            # Check if model already exists in config
            if model_id in existing_model_ids:
                duplicate_models.append(model_id)
                continue
            
            # Generate model name and costs for preview
            if custom_model_name and len(model_ids) == 1:
                model_name = custom_model_name
            else:
                model_name = self.generate_model_name(model_id)
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1
            
            input_cost = model_info.get('input_cost')
            output_cost = model_info.get('output_cost')
            
            # Handle free models
            if input_cost is not None:
                input_cost = 1e-09 if input_cost == 0.0 else input_cost
            if output_cost is not None:
                output_cost = 1e-09 if output_cost == 0.0 else output_cost
            
            valid_models.append({
                'id': model_id,
                'name': model_name,
                'input_cost': input_cost,
                'output_cost': output_cost
            })
            existing_names.append(model_name)  # Update to avoid conflicts in preview
        
        # Display preview results
        if invalid_models:
            self.logger.error(f"[DRY-RUN] {len(invalid_models)} model(s) not found in Novita API: {', '.join(invalid_models)}")
        
        if duplicate_models:
            self.logger.warning(f"[DRY-RUN] {len(duplicate_models)} model(s) already exist in configuration: {', '.join(duplicate_models)}")
        
        if valid_models:
            self.logger.info(f"[DRY-RUN] Would add {len(valid_models)} model(s):")
            for model in valid_models:
                self.logger.info(f"[DRY-RUN]   - Model '{model['id']}' with name '{model['name']}'")
                if model['input_cost'] is not None:
                    self.logger.info(f"[DRY-RUN]     Input cost: {model['input_cost']}")
                if model['output_cost'] is not None:
                    self.logger.info(f"[DRY-RUN]     Output cost: {model['output_cost']}")
        else:
            self.logger.info("[DRY-RUN] No valid models to add.")
    
    def run(self, add_models: Optional[List[str]] = None, custom_model_name: Optional[str] = None) -> int:
        """Main execution method."""
        try:
            # Load configuration
            config = self.load_config()
            
            # Handle add-models functionality
            if add_models:
                # Fetch available models with pricing from API
                available_models = self.fetch_available_models()
                
                if self.dry_run:
                    self.preview_add_model(add_models, available_models, custom_model_name)
                    return 0
                else:
                    updated_config, added_models = self.add_model_to_config(config, add_models, available_models, custom_model_name)
                    if added_models:
                        # Sort the model list after adding new models
                        updated_config, was_sorted = self.sort_model_list(updated_config)
                        self.save_config(updated_config)
                        self.logger.info(f"✅ Successfully added {len(added_models)} model(s): {', '.join(added_models)}")
                        if was_sorted:
                            self.logger.info("✅ Model list sorted by model_name")
                    else:
                        self.logger.warning("⚠️ No models were added - all models either already exist or were not found in API")
                    return 0
            
            # Sort the model list first
            updated_config, was_sorted = self.sort_model_list(config)
            
            # Fetch available models with pricing from API
            available_models = self.fetch_available_models()
            
            # Extract Novita models for cleanup/validation
            novita_models = self.extract_novita_models(updated_config)
            
            if not novita_models:
                if self.dry_run:
                    self.preview_sort_changes(config)
                    self.generate_report([], [], was_sorted)
                else:
                    if was_sorted:
                        self.save_config(updated_config)
                    self.generate_report([], [], was_sorted)
                return 0
            
            # Validate models
            invalid_models = self.validate_models(novita_models, available_models)
            
            # Validate and update costs
            updated_config, cost_changes = self.validate_and_update_costs(updated_config, novita_models, available_models)
            
            if self.dry_run:
                # Preview mode - show what would be changed
                self.preview_sort_changes(config)
                self.preview_changes(invalid_models)
                self.preview_cost_changes(cost_changes)
            else:
                # Actually apply changes
                changes_made = was_sorted
                
                # Remove invalid entries
                if invalid_models:
                    updated_config = self.remove_invalid_entries(updated_config, invalid_models)
                    changes_made = True
                
                # Save config if any changes were made (sorting, invalid models, or cost updates)
                if was_sorted or invalid_models or cost_changes:
                    self.save_config(updated_config)
                    changes_made = True
                
                if not changes_made:
                    self.logger.info("No changes needed - all models are valid with current costs and list is already sorted")
            
            # Generate final report
            self.generate_report(invalid_models, cost_changes, was_sorted)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            return 1


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean up invalid Novita models, update costs, sort model list, and add new models in LiteLLM configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs four main functions:
1. Sorts the model list alphabetically by model_name, then by litellm_params.model
2. Validates Novita models against the current API and removes invalid entries
3. Updates model costs (input_cost_per_token/output_cost_per_token) when they differ from API pricing
4. Adds one or more Novita models to the configuration

Examples:
  %(prog)s                           # Process config.yaml (sort + validate models + update costs)
  %(prog)s --config my-config.yaml   # Process custom config file
  %(prog)s --dry-run                 # Preview all changes without modifying file
  %(prog)s --verbose --dry-run       # Detailed preview mode with debug information
  %(prog)s --add-model "deepseek/deepseek-r1-0528-qwen3-8b"  # Add a single model
  %(prog)s --add-model "deepseek/deepseek-r1-0528-qwen3-8b" "qwen/qwen-2.5-72b-instruct"  # Add multiple models
  %(prog)s --add-model "qwen/qwen-2.5-72b-instruct" --dry-run  # Preview adding a model
        """
    )
    
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_FILE,
        help=f'Path to LiteLLM configuration file (default: {DEFAULT_CONFIG_FILE})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview all changes (sorting, model removals, and cost updates) without modifying the configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output with detailed cost comparison information'
    )
    
    parser.add_argument(
        '--add-model',
        nargs='*',
        help='Add one or more Novita models to the configuration. Provide model IDs separated by spaces or use quotes for each model'
    )

    parser.add_argument(
        '--model-name',
        help='Custom model name to use when adding a single model. Only valid when --add-model is used with exactly one model.'
    )
    
    args = parser.parse_args()
    
    # Process the add_models argument to handle both space-separated and quoted models
    processed_add_models = None
    if args.add_model:
        # Flatten the list by splitting any space-separated models
        processed_add_models = []
        for item in args.add_model:
            # Split by spaces to handle space-separated models
            models = item.split()
            processed_add_models.extend(models)

    # Validate --model-name usage
    if args.model_name:
        if not processed_add_models:
            parser.error("--model-name can only be used with --add-model")
        if len(processed_add_models) > 1:
            parser.error("--model-name can only be used when adding a single model")
    
    # Create and run the cleaner
    cleaner = NovitaModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run(add_models=processed_add_models, custom_model_name=args.model_name)


if __name__ == '__main__':
    sys.exit(main())