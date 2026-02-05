#!/usr/bin/env python3
"""
Synthetic Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Synthetic models in a LiteLLM config.yaml file against
the current Synthetic API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token)

Usage:
    python cleanup_synthetic_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_synthetic_models.py --add-model "hf:zai-org/GLM-4.7"

Author: Generated for LiteLLM Config Management
"""

import argparse
import os
import sys
import requests
from typing import Dict, List, Tuple, Any, Optional

from cleanup_base import (
    BaseModelCleaner,
    setup_common_args,
    validate_model_name_arg,
    fetch_models_from_api,
    adjust_cost_for_free_model,
)
from typing import Set


SYNTHETIC_API_URL = "https://api.synthetic.new/openai/v1/models"


class SyntheticModelCleaner(BaseModelCleaner):
    """Cleaner for Synthetic models."""
    
    PROVIDER_NAME = "Synthetic"
    API_URL = SYNTHETIC_API_URL
    MODEL_PREFIX = "openai/"
    SPECIAL_MODELS: Set[str] = {"hf:nomic-ai/nomic-embed-text-v1.5"}
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        super().__init__(config_path, dry_run, verbose)
        self.api_key = os.environ.get('SYNTHETIC_API_KEY')
        self.api_base = os.environ.get('SYNTHETIC_API_BASE', 'https://api.synthetic.com/v1')
    
    def extract_provider_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """Extract Synthetic models from the configuration."""
        synthetic_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
            
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            api_base = str(litellm_params.get('api_base', ''))
            
            # Check if it's a synthetic model by api_base containing SYNTHETIC_API_BASE
            is_synthetic = 'SYNTHETIC_API_BASE' in api_base
            
            if is_synthetic and model_id.startswith(self.MODEL_PREFIX):
                model_name = model_entry.get('model_name', 'unnamed')
                synthetic_models.append((index, model_id, model_name))
                self.logger.debug(f"Found Synthetic model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(synthetic_models)} Synthetic models in configuration")
        return synthetic_models

    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch available models with pricing from Synthetic API."""
        try:
            available_models = {}
            headers = {}
            
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.get(self.API_URL, headers=headers if headers else None, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for model in data.get('data', []):
                if isinstance(model, dict) and 'id' in model:
                    model_id = model['id']
                    model_info = {
                        'id': model_id,
                        'input_cost': None,
                        'output_cost': None,
                        'model_info': None
                    }
                    
                    pricing = model.get('pricing', {})
                    if isinstance(pricing, dict):
                        prompt_cost = pricing.get('prompt')
                        if prompt_cost is not None:
                            try:
                                # Parse string values like "$0.00000055"
                                if isinstance(prompt_cost, str):
                                    prompt_cost = prompt_cost.replace('$', '').replace(',', '')
                                model_info['input_cost'] = float(prompt_cost)
                            except (ValueError, TypeError):
                                self.logger.debug(f"Invalid prompt cost for {model_id}: {prompt_cost}")

                        completion_cost = pricing.get('completion')
                        if completion_cost is not None:
                            try:
                                # Parse string values like "$0.00000219"
                                if isinstance(completion_cost, str):
                                    completion_cost = completion_cost.replace('$', '').replace(',', '')
                                model_info['output_cost'] = float(completion_cost)
                            except (ValueError, TypeError):
                                self.logger.debug(f"Invalid completion cost for {model_id}: {completion_cost}")
                    
                    available_models[model_id] = model_info
            
            self.logger.info(f"Fetched {len(available_models)} available models from Synthetic API")
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error fetching models from Synthetic API: {e}")
            raise
    
    def get_api_model_id(self, model_id: str) -> str:
        """Extract the API model ID from the config model ID."""
        if model_id.startswith(self.MODEL_PREFIX):
            return model_id[len(self.MODEL_PREFIX):]
        return model_id
    
    def generate_model_name(self, model_id: str) -> str:
        """Generate model name for Synthetic models (no prefix, cleanup ID)."""
        # Remove hf: and zai-org/ prefixes
        clean_id = model_id.replace('hf:', '').replace('zai-org/', '')
        # Replace / and : with -
        clean_id = clean_id.replace('/', '-').replace(':', '-')
        return clean_id.lower()
    
    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any],
                          model_name: str) -> Dict[str, Any]:
        """Create a new Synthetic model entry for the config."""
        input_cost = adjust_cost_for_free_model(api_model_info.get('input_cost'))
        output_cost = adjust_cost_for_free_model(api_model_info.get('output_cost'))
        
        entry = {
            'model_name': model_name,
            'litellm_params': {
                'model': f'{self.MODEL_PREFIX}{model_id}',
                'api_base': 'os.environ/SYNTHETIC_API_BASE',
                'api_key': 'os.environ/SYNTHETIC_API_KEY',
            }
        }
        
        if input_cost is not None:
            entry['litellm_params']['input_cost_per_token'] = input_cost
        if output_cost is not None:
            entry['litellm_params']['output_cost_per_token'] = output_cost
        
        model_info = api_model_info.get('model_info')
        if model_info:
            entry['model_info'] = model_info
        
        return entry
    
    def run_cleanup(self, add_models: Optional[List[str]] = None,
                   custom_model_name: Optional[str] = None) -> int:
        """Run the complete cleanup process."""
        try:
            config = self.load_config()
            api_models = self.fetch_available_models()
            
            if add_models:
                if self.dry_run:
                    self.preview_add_model(add_models, api_models, custom_model_name)
                    return 0
                else:
                    config, added_models = self.add_model_to_config(
                        config, add_models, api_models, custom_model_name)
                    if added_models:
                        config, was_sorted = self.sort_model_list(config)
                        self.save_config(config)
                        self.logger.info(f"✅ Successfully added {len(added_models)} model(s)")
                    else:
                        self.logger.warning("⚠️ No models were added")
                    return 0
            
            config_models = self.extract_provider_models(config)
            invalid_models = self.validate_models(config_models, api_models)
            config, cost_changes = self.validate_and_update_costs(config, config_models, api_models)
            config, was_sorted = self.sort_model_list(config)
            
            if self.dry_run:
                self.preview_sort_changes(config)
                self.preview_changes(invalid_models)
                self.preview_cost_changes(cost_changes)
            else:
                if invalid_models:
                    config = self.remove_invalid_entries(config, invalid_models)
                
                if invalid_models or cost_changes or was_sorted:
                    self.save_config(config)
            
            self.generate_report(invalid_models, cost_changes, was_sorted)
            return 0
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup Synthetic models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model "hf:zai-org/GLM-4.7"  # Add new model(s)
        """
    )
    
    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)
    
    cleaner = SyntheticModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
