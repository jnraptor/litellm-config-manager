#!/usr/bin/env python3
"""
Nvidia Model Cleanup Script for LiteLLM Config

This script validates Nvidia models in a LiteLLM config.yaml file against
the current Nvidia API and:
1. Removes any invalid model entries
2. Adds new models when requested

The script fetches current models from https://integrate.api.nvidia.com/v1/models

Note: Nvidia models are free, so all models use a nominal cost of 1e-09 for LiteLLM compatibility.

Usage:
    python cleanup_nvidia_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_nvidia_models.py --add-model "meta/llama-3.1-8b-instruct"

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
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/models"
DEFAULT_CONFIG_FILE = "config.yaml"
# Nominal cost for free models (LiteLLM requires non-zero costs)
FREE_MODEL_COST = 1.0e-09


class NvidiaModelCleaner:
    """Main class for cleaning up invalid Nvidia models from LiteLLM config."""
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
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
    
    def extract_nvidia_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """Extract Nvidia models from the configuration."""
        nvidia_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
                
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            model_name = model_entry.get('model_name', 'unnamed')
            
            if model_id.startswith('nvidia_nim/'):
                nvidia_models.append((index, model_id, model_name))
                self.logger.debug(f"Found Nvidia model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(nvidia_models)} Nvidia models in configuration")
        return nvidia_models
    
    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch the list of available models from Nvidia API."""
        try:
            self.logger.info("Fetching available models from Nvidia API...")
            
            response = requests.get(NVIDIA_API_URL, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("Invalid API response format: missing 'data' field")
            
            available_models = {}
            for model in data['data']:
                if isinstance(model, dict) and 'id' in model:
                    model_id = model['id']
                    model_info = {
                        'id': model_id,
                        'input_cost': FREE_MODEL_COST,
                        'output_cost': FREE_MODEL_COST,
                        'owned_by': model.get('owned_by', 'unknown')
                    }
                    available_models[model_id] = model_info
            
            self.logger.info(f"Fetched {len(available_models)} available models from Nvidia API")
            self.logger.debug(f"Sample models: {list(available_models.keys())[:5]}")
            
            return available_models
            
        except requests.RequestException as e:
            self.logger.error(f"Error fetching models from Nvidia API: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing API response: {e}")
            raise
    
    def validate_models(self, config_models: List[Tuple[int, str, str]],
                       api_models: Dict[str, Dict[str, Any]]) -> List[Tuple[int, str, str]]:
        """Compare config models with API models and identify invalid ones."""
        api_models_set = set(api_models.keys())
        invalid_models = []
        
        for index, model_id, model_name in config_models:
            if model_id.startswith('nvidia_nim/'):
                api_model_id = model_id[len('nvidia_nim/'):]
                
                if api_model_id not in api_models_set:
                    invalid_models.append((index, model_id, model_name))
                    self.logger.debug(f"Invalid model found: {model_id} -> {api_model_id}")
                else:
                    self.logger.debug(f"Valid model: {model_id} -> {api_model_id}")
        
        self.logger.info(f"Identified {len(invalid_models)} invalid Nvidia models")
        return invalid_models
    
    def preview_changes(self, invalid_models: List[Tuple[int, str, str]]) -> None:
        """Preview what changes would be made in dry-run mode."""
        if not invalid_models:
            self.logger.info("[DRY-RUN] No invalid Nvidia models found. No changes needed.")
            return
        
        self.logger.info("[DRY-RUN] Would remove the following invalid Nvidia models:")
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
    
    def sort_model_list(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Sort the model list by model_name alphabetically."""
        if 'model_list' not in config or not config['model_list']:
            self.logger.info("No model list found or model list is empty")
            return config, False
        
        model_list = config['model_list']
        
        original_order = [(m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', '')) 
                         for m in model_list]
        
        sorted_model_list = sorted(
            model_list,
            key=lambda x: (
                x.get('model_name', 'unnamed').lower(),
                x.get('litellm_params', {}).get('model', '').lower()
            )
        )
        
        sorted_order = [(m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', '')) 
                       for m in sorted_model_list]
        
        was_sorted = original_order != sorted_order
        
        if was_sorted:
            config['model_list'] = sorted_model_list
            self.logger.info(f"Sorted {len(model_list)} models by model_name")
        else:
            self.logger.info("Model list is already sorted")
        
        return config, was_sorted
    
    def preview_sort_changes(self, config: Dict[str, Any]) -> None:
        """Preview what the sorting would change in dry-run mode."""
        if 'model_list' not in config or not config['model_list']:
            self.logger.info("[DRY-RUN] No model list found or model list is empty")
            return
        
        model_list = config['model_list']
        original_order = [(m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', '')) 
                         for m in model_list]
        
        sorted_model_list = sorted(
            model_list,
            key=lambda x: (
                x.get('model_name', 'unnamed').lower(),
                x.get('litellm_params', {}).get('model', '').lower()
            )
        )
        
        sorted_order = [(m.get('model_name', 'unnamed'), m.get('litellm_params', {}).get('model', '')) 
                       for m in sorted_model_list]
        
        would_sort = original_order != sorted_order
        
        if would_sort:
            self.logger.info(f"[DRY-RUN] Would sort {len(model_list)} models by model_name")
        else:
            self.logger.info("[DRY-RUN] Model list is already sorted - no changes needed")

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
                       was_sorted: bool = False) -> None:
        """Generate a summary report of the cleanup operation."""
        if invalid_models:
            if self.dry_run:
                self.logger.info(f"📋 [DRY-RUN] Summary: {len(invalid_models)} invalid models identified")
            else:
                self.logger.info(f"✅ Model cleanup: {len(invalid_models)} invalid models removed")
                
            for _, model_id, model_name in invalid_models:
                status = "[WOULD REMOVE]" if self.dry_run else "[REMOVED]"
                self.logger.info(f"  {status} {model_id} (name: {model_name})")
        
        if was_sorted:
            if self.dry_run:
                self.logger.info("📝 [DRY-RUN] Model list would be sorted by model_name")
            else:
                self.logger.info("✅ Model list sorted by model_name")
        
        if not invalid_models and not was_sorted:
            self.logger.info("✅ All Nvidia models are valid and list is already sorted")
        elif self.dry_run:
            total_changes = len(invalid_models) + (1 if was_sorted else 0)
            self.logger.info(f"📋 [DRY-RUN] Total changes identified: {total_changes}")
            self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
        else:
            total_changes = len(invalid_models) + (1 if was_sorted else 0)
            self.logger.info(f"✅ Cleanup completed: {total_changes} total changes applied")
    
    def generate_model_name(self, model_id: str) -> str:
        """Generate an appropriate model_name from the Nvidia model ID."""
        clean_id = model_id.replace('/', '-').replace(':', '-')
        model_name = f"nim-{clean_id}"
        return model_name.lower()
    
    def find_model_in_api(self, model_id: str, api_models: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find a specific model in the API models data."""
        if model_id in api_models:
            return api_models[model_id]
        return None
    
    def add_model_to_config(self, config: Dict[str, Any], model_ids: List[str], 
                           api_models: Dict[str, Dict[str, Any]], 
                           custom_model_name: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """Add one or more Nvidia models to the configuration."""
        added_models = []
        failed_models = []
        
        existing_models = self.extract_nvidia_models(config)
        existing_model_ids = [mid.replace('nvidia_nim/', '') for _, mid, _ in existing_models]
        existing_names = [name for _, _, name in existing_models]
        
        for model_id in model_ids:
            self.logger.info(f"Processing model: {model_id}")
            
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                self.logger.error(f"Model '{model_id}' not found in Nvidia API")
                failed_models.append(model_id)
                continue
            
            if model_id in existing_model_ids:
                self.logger.warning(f"Model '{model_id}' already exists in configuration")
                failed_models.append(model_id)
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
            
            model_entry = {
                'litellm_params': {
                    'model': f'nvidia_nim/{model_id}',
                    'input_cost_per_token': FREE_MODEL_COST,
                    'output_cost_per_token': FREE_MODEL_COST
                },
                'model_name': model_name
            }
            
            config['model_list'].append(model_entry)
            added_models.append(model_id)
            existing_model_ids.append(model_id)
            existing_names.append(model_name)
            
            self.logger.info(f"Added model '{model_id}' with name '{model_name}'")
            self.logger.info(f"  Input cost: {FREE_MODEL_COST} (free)")
            self.logger.info(f"  Output cost: {FREE_MODEL_COST} (free)")
        
        if failed_models:
            self.logger.warning(f"Failed to add {len(failed_models)} model(s): {', '.join(failed_models)}")
        if added_models:
            self.logger.info(f"Successfully processed {len(added_models)} model(s) for addition")
        
        return config, added_models
    
    def preview_add_model(self, model_ids: List[str], api_models: Dict[str, Dict[str, Any]], 
                         custom_model_name: Optional[str] = None) -> None:
        """Preview what would be added when adding one or more models."""
        config = self.load_config()
        existing_models = self.extract_nvidia_models(config)
        existing_model_ids = [mid.replace('nvidia_nim/', '') for _, mid, _ in existing_models]
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
                'input_cost': FREE_MODEL_COST,
                'output_cost': FREE_MODEL_COST
            }
            
            valid_models.append(model_preview)
            existing_names.append(model_name)
        
        if invalid_models:
            self.logger.error(f"[DRY-RUN] {len(invalid_models)} model(s) not found in Nvidia API: {', '.join(invalid_models)}")
        
        if duplicate_models:
            self.logger.warning(f"[DRY-RUN] {len(duplicate_models)} model(s) already exist in configuration: {', '.join(duplicate_models)}")
        
        if valid_models:
            self.logger.info(f"[DRY-RUN] Would add {len(valid_models)} model(s):")
            for model in valid_models:
                self.logger.info(f"[DRY-RUN]   - Model '{model['id']}' with name '{model['name']}'")
                self.logger.info(f"[DRY-RUN]     Input cost: {model['input_cost']} (free)")
                self.logger.info(f"[DRY-RUN]     Output cost: {model['output_cost']} (free)")
        else:
            self.logger.info("[DRY-RUN] No valid models to add.")
    
    def run(self, add_models: Optional[List[str]] = None, custom_model_name: Optional[str] = None) -> int:
        """Main execution method."""
        try:
            config = self.load_config()
            
            if add_models:
                available_models = self.fetch_available_models()
                
                if self.dry_run:
                    self.preview_add_model(add_models, available_models, custom_model_name)
                    return 0
                else:
                    updated_config, added_models = self.add_model_to_config(
                        config, add_models, available_models, custom_model_name)
                    if added_models:
                        updated_config, was_sorted = self.sort_model_list(updated_config)
                        self.save_config(updated_config)
                        self.logger.info(f"✅ Successfully added {len(added_models)} model(s): {', '.join(added_models)}")
                        if was_sorted:
                            self.logger.info("✅ Model list sorted by model_name")
                    else:
                        self.logger.warning("⚠️ No models were added")
                    return 0
            
            updated_config, was_sorted = self.sort_model_list(config)
            available_models = self.fetch_available_models()
            nvidia_models = self.extract_nvidia_models(updated_config)
            
            if not nvidia_models:
                if self.dry_run:
                    self.preview_sort_changes(config)
                    self.generate_report([], was_sorted)
                else:
                    if was_sorted:
                        self.save_config(updated_config)
                    self.generate_report([], was_sorted)
                return 0
            
            invalid_models = self.validate_models(nvidia_models, available_models)
            
            if self.dry_run:
                self.preview_sort_changes(config)
                self.preview_changes(invalid_models)
            else:
                changes_made = was_sorted
                
                if invalid_models:
                    updated_config = self.remove_invalid_entries(updated_config, invalid_models)
                    changes_made = True
                
                if was_sorted or invalid_models:
                    self.save_config(updated_config)
                    changes_made = True
                
                if not changes_made:
                    self.logger.info("No changes needed - all models are valid and list is already sorted")
            
            self.generate_report(invalid_models, was_sorted)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Script execution failed: {e}")
            return 1


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean up invalid Nvidia models and add new models in LiteLLM configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs the following functions:
1. Sorts the model list alphabetically by model_name
2. Validates Nvidia models against the current API and removes invalid entries
3. Adds one or more Nvidia models to the configuration

Note: All Nvidia models are free, so they use a nominal cost of 1e-09 for LiteLLM compatibility.

Examples:
  %(prog)s                                              # Process config.yaml
  %(prog)s --config my-config.yaml                      # Process custom config file
  %(prog)s --dry-run                                    # Preview all changes
  %(prog)s --add-model "meta/llama-3.1-8b-instruct"     # Add a single model
  %(prog)s --add-model "meta/llama-3.1-8b-instruct" "nvidia/nemotron-mini-4b-instruct"  # Add multiple
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
        help='Preview all changes without modifying the configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--add-model',
        nargs='*',
        help='Add one or more Nvidia models to the configuration'
    )

    parser.add_argument(
        '--model-name',
        help='Custom model name (only valid when adding a single model)'
    )
    
    args = parser.parse_args()
    
    processed_add_models = None
    if args.add_model:
        processed_add_models = []
        for item in args.add_model:
            models = item.split()
            processed_add_models.extend(models)

    if args.model_name:
        if not processed_add_models:
            parser.error("--model-name can only be used with --add-model")
        if len(processed_add_models) > 1:
            parser.error("--model-name can only be used when adding a single model")
    
    cleaner = NvidiaModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run(add_models=processed_add_models, custom_model_name=args.model_name)


if __name__ == '__main__':
    sys.exit(main())
