#!/usr/bin/env python3
"""
Nvidia Model Cleanup Script for LiteLLM Config

This script validates Nvidia models in a LiteLLM config.yaml file against
the current Nvidia API and:
1. Removes any invalid model entries
2. Adds new models when requested

Note: Nvidia models are free, so all models use a nominal cost of 1e-09 for LiteLLM compatibility.

Usage:
    python cleanup_nvidia_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_nvidia_models.py --add-model "meta/llama-3.1-8b-instruct"

Author: Generated for LiteLLM Config Management
"""

import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional

from cleanup_base import (
    BaseModelCleaner,
    setup_common_args,
    validate_model_name_arg,
    fetch_models_from_api,
)


NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/models"
FREE_MODEL_COST = 1.0e-09


class NvidiaModelCleaner(BaseModelCleaner):
    """Cleaner for Nvidia NIM models."""
    
    PROVIDER_NAME = "Nvidia"
    API_URL = NVIDIA_API_URL
    MODEL_PREFIX = "nvidia_nim/"
    
    def extract_provider_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """Extract Nvidia models from the configuration."""
        nvidia_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
            
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            
            if model_id.startswith(self.MODEL_PREFIX):
                model_name = model_entry.get('model_name', 'unnamed')
                nvidia_models.append((index, model_id, model_name))
                self.logger.debug(f"Found Nvidia model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(nvidia_models)} Nvidia models in configuration")
        return nvidia_models

    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch available models from Nvidia API."""
        try:
            data = fetch_models_from_api(self.API_URL, self.logger)
            
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
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error fetching models from Nvidia API: {e}")
            raise
    
    def get_api_model_id(self, model_id: str) -> str:
        """Extract the API model ID from the config model ID."""
        return model_id.replace(self.MODEL_PREFIX, '')
    
    def generate_model_name(self, model_id: str, prefix: str = "nim-") -> str:
        """Generate model name with nim- prefix for Nvidia models."""
        return super().generate_model_name(model_id, prefix)
    
    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any],
                          model_name: str) -> Dict[str, Any]:
        """Create a new Nvidia model entry for the config."""
        return {
            'model_name': model_name,
            'litellm_params': {
                'model': f'{self.MODEL_PREFIX}{model_id}',
                'input_cost_per_token': FREE_MODEL_COST,
                'output_cost_per_token': FREE_MODEL_COST
            }
        }
    
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
            config, was_sorted = self.sort_model_list(config)
            
            if not config_models:
                if self.dry_run:
                    self.preview_sort_changes(config)
                else:
                    if was_sorted:
                        self.save_config(config)
                self.generate_report([], None, was_sorted)
                return 0
            
            invalid_models = self.validate_models(config_models, api_models)
            
            if self.dry_run:
                self.preview_sort_changes(config)
                self.preview_changes(invalid_models)
            else:
                if invalid_models:
                    config = self.remove_invalid_entries(config, invalid_models)
                
                if invalid_models or was_sorted:
                    self.save_config(config)
            
            self.generate_report(invalid_models, None, was_sorted)
            return 0
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup Nvidia NIM models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: All Nvidia models are free, so they use a nominal cost of 1e-09 for LiteLLM compatibility.

Examples:
  %(prog)s                                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml                             # Run cleanup on custom config file
  %(prog)s --dry-run                                    # Preview changes without modifying file
  %(prog)s --add-model "meta/llama-3.1-8b-instruct"     # Add a single model
        """
    )
    
    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)
    
    cleaner = NvidiaModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
