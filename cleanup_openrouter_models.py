#!/usr/bin/env python3
"""
OpenRouter Model Cleanup and Cost Update Script for LiteLLM Config

This script validates OpenRouter models in a LiteLLM config.yaml file against
the current OpenRouter API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing

The script fetches current pricing from https://openrouter.ai/api/v1/models and automatically
updates any cost differences found in the configuration file.

Usage:
    python cleanup_openrouter_models.py [--config config.yaml] [--dry-run] [--verbose]

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
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CONFIG_FILE = "config.yaml"


class OpenRouterModelCleaner:
    """Main class for cleaning up invalid OpenRouter models from LiteLLM config."""
    
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
    
    def extract_openrouter_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """
        Extract OpenRouter models from the configuration.
        
        Returns:
            List of tuples: (index, model_id, model_name)
        """
        openrouter_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
                
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            model_name = model_entry.get('model_name', 'unnamed')
            
            # Check if this is an OpenRouter model
            if model_id.startswith('openrouter/'):
                openrouter_models.append((index, model_id, model_name))
                self.logger.debug(f"Found OpenRouter model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(openrouter_models)} OpenRouter models in configuration")
        return openrouter_models
    
    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch the list of available models with pricing from OpenRouter API."""
        try:
            self.logger.info("Fetching available models with pricing from OpenRouter API...")
            
            response = requests.get(OPENROUTER_API_URL, timeout=30)
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
                    
                    # Extract pricing information
                    pricing = model.get('pricing', {})
                    if isinstance(pricing, dict):
                        # Parse input cost (prompt)
                        prompt_cost = pricing.get('prompt')
                        if prompt_cost is not None:
                            try:
                                model_info['input_cost'] = float(prompt_cost)
                            except (ValueError, TypeError):
                                self.logger.debug(f"Invalid prompt cost for {model_id}: {prompt_cost}")
                        
                        # Parse output cost (completion)
                        completion_cost = pricing.get('completion')
                        if completion_cost is not None:
                            try:
                                model_info['output_cost'] = float(completion_cost)
                            except (ValueError, TypeError):
                                self.logger.debug(f"Invalid completion cost for {model_id}: {completion_cost}")
                    
                    available_models[model_id] = model_info
            
            self.logger.info(f"Fetched {len(available_models)} available models from OpenRouter API")
            self.logger.debug(f"Sample models: {list(available_models.keys())[:5]}")
            
            return available_models
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching models from OpenRouter API: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing API response: {e}")
            raise
    
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
            # Extract the model ID part after 'openrouter/'
            # OpenRouter models in config are like: openrouter/qwen/qwen3-14b:free
            # API models are like: qwen/qwen3-14b:free
            if model_id.startswith('openrouter/'):
                api_model_id = model_id[len('openrouter/'):]  # Remove 'openrouter/' prefix
                
                if api_model_id not in api_models_set:
                    invalid_models.append((index, model_id, model_name))
                    self.logger.debug(f"Invalid model found: {model_id} -> {api_model_id}")
                else:
                    self.logger.debug(f"Valid model: {model_id} -> {api_model_id}")
        
        self.logger.info(f"Identified {len(invalid_models)} invalid OpenRouter models")
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
            if model_id.startswith('openrouter/'):
                api_model_id = model_id[len('openrouter/'):]
                
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
                        
                        if current_input_cost != adjusted_input_cost:
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
                        
                        if current_output_cost != adjusted_output_cost:
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
            self.logger.info("[DRY-RUN] No invalid OpenRouter models found. No changes needed.")
            return
        
        self.logger.info("[DRY-RUN] Would remove the following invalid OpenRouter models:")
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
                       cost_changes: Optional[List[Dict[str, Any]]] = None) -> None:
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
        
        # Overall summary
        if not invalid_models and not cost_changes:
            self.logger.info("✅ All OpenRouter models are valid with current costs")
        elif self.dry_run:
            total_changes = len(invalid_models) + len(cost_changes)
            self.logger.info(f"📋 [DRY-RUN] Total changes identified: {total_changes}")
            self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
        else:
            total_changes = len(invalid_models) + len(cost_changes)
            self.logger.info(f"✅ Cleanup completed: {total_changes} total changes applied")
    
    def generate_model_name(self, model_id: str) -> str:
        """
        Generate an appropriate model_name from the OpenRouter model ID.
        
        Args:
            model_id: The OpenRouter model ID (e.g., "anthropic/claude-3-5-sonnet-20241022")
            
        Returns:
            Generated model name (e.g., "or-claude-3-5-sonnet-20241022")
        """
        # Remove common prefixes and clean up the name
        clean_id = model_id.replace('/', '-').replace(':', '-')
        
        # Add 'or-' prefix for OpenRouter models
        model_name = f"or-{clean_id}"
        
        # Handle some common cases to make names more readable
        model_name = model_name.replace('anthropic-', '').replace('meta-llama-', '').replace('google-', '')
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
        
        # Also check for variations (with/without :free suffix)
        if f"{model_id}:free" in api_models:
            return api_models[f"{model_id}:free"]
        
        # Check if the provided ID has :free and try without it
        if model_id.endswith(':free'):
            base_id = model_id[:-5]  # Remove ':free'
            if base_id in api_models:
                return api_models[base_id]
        
        return None
    
    def add_model_to_config(self, config: Dict[str, Any], model_id: str, 
                           api_models: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Add a new OpenRouter model to the configuration.
        
        Args:
            config: The configuration dictionary
            model_id: The OpenRouter model ID to add
            api_models: Dict of model data from API
            
        Returns:
            Tuple of (updated_config, list_of_added_models)
        """
        added_models = []
        
        # Find the model in API data
        model_info = self.find_model_in_api(model_id, api_models)
        if not model_info:
            self.logger.error(f"Model '{model_id}' not found in OpenRouter API")
            return config, added_models
        
        # Check if model already exists in config
        existing_models = self.extract_openrouter_models(config)
        existing_model_ids = [mid.replace('openrouter/', '') for _, mid, _ in existing_models]
        
        if model_id in existing_model_ids:
            self.logger.warning(f"Model '{model_id}' already exists in configuration")
            return config, added_models
        
        # Generate model name
        model_name = self.generate_model_name(model_id)
        
        # Check for name conflicts and make unique if needed
        existing_names = [name for _, _, name in existing_models]
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
                'model': f'openrouter/{model_id}'
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
        
        self.logger.info(f"Added model '{model_id}' with name '{model_name}'")
        if input_cost is not None:
            self.logger.info(f"  Input cost: {input_cost}")
        if output_cost is not None:
            self.logger.info(f"  Output cost: {output_cost}")
        
        # Also add free version if paid version exists and free version is available
        free_model_id = f"{model_id}:free"
        if not model_id.endswith(':free') and free_model_id in api_models:
            free_model_info = api_models[free_model_id]
            # Use the same model name for both free and paid versions
            free_model_name = model_name
            
            free_model_entry = {
                'litellm_params': {
                    'model': f'openrouter/{free_model_id}',
                    'input_cost_per_token': 1e-09,
                    'output_cost_per_token': 1e-09
                },
                'model_name': free_model_name
            }
            
            config['model_list'].append(free_model_entry)
            added_models.append(free_model_id)
            
            self.logger.info(f"Also added free version '{free_model_id}' with same name '{free_model_name}'")
        
        return config, added_models
    
    def preview_add_model(self, model_id: str, api_models: Dict[str, Dict[str, Any]]) -> None:
        """Preview what would be added when adding a new model."""
        model_info = self.find_model_in_api(model_id, api_models)
        if not model_info:
            self.logger.error(f"[DRY-RUN] Model '{model_id}' not found in OpenRouter API")
            return
        
        # Check if model already exists in config (same logic as add_model_to_config)
        config = self.load_config()
        existing_models = self.extract_openrouter_models(config)
        existing_model_ids = [mid.replace('openrouter/', '') for _, mid, _ in existing_models]
        
        if model_id in existing_model_ids:
            self.logger.warning(f"[DRY-RUN] Model '{model_id}' already exists in configuration")
            return
        
        model_name = self.generate_model_name(model_id)
        input_cost = model_info.get('input_cost')
        output_cost = model_info.get('output_cost')
        
        # Handle free models
        if input_cost is not None:
            input_cost = 1e-09 if input_cost == 0.0 else input_cost
        if output_cost is not None:
            output_cost = 1e-09 if output_cost == 0.0 else output_cost
        
        self.logger.info(f"[DRY-RUN] Would add model '{model_id}' with name '{model_name}'")
        if input_cost is not None:
            self.logger.info(f"[DRY-RUN]   Input cost: {input_cost}")
        if output_cost is not None:
            self.logger.info(f"[DRY-RUN]   Output cost: {output_cost}")
        
        # Check for free version
        free_model_id = f"{model_id}:free"
        if not model_id.endswith(':free') and free_model_id in api_models:
            # Use the same model name for both free and paid versions
            free_model_name = model_name
            self.logger.info(f"[DRY-RUN] Would also add free version '{free_model_id}' with same name '{free_model_name}'")
    
    def run(self, add_model: Optional[str] = None) -> int:
        """Main execution method."""
        try:
            # Load configuration
            config = self.load_config()
            
            # Fetch available models with pricing from API
            available_models = self.fetch_available_models()
            
            # Handle add-model functionality
            if add_model:
                if self.dry_run:
                    self.preview_add_model(add_model, available_models)
                    return 0
                else:
                    updated_config, added_models = self.add_model_to_config(config, add_model, available_models)
                    if added_models:
                        self.save_config(updated_config)
                        self.logger.info(f"✅ Successfully added {len(added_models)} model(s): {', '.join(added_models)}")
                    return 0
            
            # Extract OpenRouter models for cleanup/validation
            openrouter_models = self.extract_openrouter_models(config)
            
            if not openrouter_models:
                self.logger.info("No OpenRouter models found in configuration")
                return 0
            
            # Validate models
            invalid_models = self.validate_models(openrouter_models, available_models)
            
            # Validate and update costs
            updated_config, cost_changes = self.validate_and_update_costs(config, openrouter_models, available_models)
            
            if self.dry_run:
                # Preview mode - show what would be changed
                self.preview_changes(invalid_models)
                self.preview_cost_changes(cost_changes)
            else:
                # Actually apply changes
                changes_made = False
                
                # Remove invalid entries
                if invalid_models:
                    updated_config = self.remove_invalid_entries(updated_config, invalid_models)
                    changes_made = True
                
                # Save config if any changes were made (invalid models or cost updates)
                if invalid_models or cost_changes:
                    self.save_config(updated_config)
                    changes_made = True
                
                if not changes_made:
                    self.logger.info("No changes needed - all models are valid with current costs")
            
            # Generate final report
            self.generate_report(invalid_models, cost_changes)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            return 1


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean up invalid OpenRouter models, update costs, and add new models in LiteLLM configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs three main functions:
1. Validates OpenRouter models against the current API and removes invalid entries
2. Updates model costs (input_cost_per_token/output_cost_per_token) when they differ from API pricing
3. Adds new OpenRouter models to the configuration

Examples:
  %(prog)s                           # Process config.yaml (validate models + update costs)
  %(prog)s --config my-config.yaml   # Process custom config file
  %(prog)s --dry-run                 # Preview all changes without modifying file
  %(prog)s --verbose --dry-run       # Detailed preview mode with debug information
  %(prog)s --add-model "anthropic/claude-3-5-sonnet-20241022"  # Add a new model
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
        help='Preview all changes (model removals and cost updates) without modifying the configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output with detailed cost comparison information'
    )
    
    parser.add_argument(
        '--add-model',
        type=str,
        help='Add a new OpenRouter model to the configuration. Provide the model ID (e.g., "anthropic/claude-3-5-sonnet-20241022")'
    )
    
    args = parser.parse_args()
    
    # Create and run the cleaner
    cleaner = OpenRouterModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run(add_model=args.add_model)


if __name__ == '__main__':
    sys.exit(main())